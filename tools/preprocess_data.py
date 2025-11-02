#!/usr/bin/env python3
"""
Generic preprocessing pipeline for IndexTTS2 datasets.

This script mirrors `tools/preprocess_japanese.py`, but exposes a configurable
`--language` flag so we can target different SentencePiece models / normaliser
settings without duplicating the implementation for each locale.

Workflow recap:
  1. Normalise and tokenize text with the provided SentencePiece model.
  2. Load audio, extract semantic features with SeamlessM4T + Wav2Vec2Bert.
  3. Quantise semantic codes using the MaskGCT semantic codec.
  4. Extract conditioning latents & emotion vectors from the UnifiedVoice GPT.
  5. Persist `.npy` artefacts and write train/validation JSONL manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct_utils import build_semantic_codec, build_semantic_model
from huggingface_hub import hf_hub_download
import safetensors.torch


def load_existing_ids(manifest_path: Path) -> set[str]:
    ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids.add(record["id"])
    return ids


def update_stats_file(
    stats_path: Path,
    train_ids: set[str],
    val_ids: set[str],
    tokenizer_path: Path,
    checkpoint_path: Path,
) -> None:
    stats = {
        "total": len(train_ids) + len(val_ids),
        "train": len(train_ids),
        "val": len(val_ids),
        "tokenizer": str(tokenizer_path),
        "gpt_checkpoint": str(checkpoint_path),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as stats_f:
        json.dump(stats, stats_f, indent=2, ensure_ascii=False)


def assign_to_validation(sample_id: str, ratio: float) -> bool:
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    value = int(digest, 16) % 1_000_000
    return (value / 1_000_000) < ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for IndexTTS2 fine-tuning."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("JA_yodas_dataset/ja_yodas_train.jsonl"),
        help="Source manifest (JSONL) with fields id/text/audio/speaker/language.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="LANG=MANIFEST[=OUTPUT]",
        help=(
            "Additional dataset to process. Provide entries like "
            "`ja=datasets/JA_yodas_dataset/ja_yodas_train.jsonl` or "
            "`ja=datasets/JA_yodas_dataset/ja_yodas_train.jsonl=ja_processed_data`."
            "Can be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data"),
        help="Directory to store processed artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for outputs when using --dataset entries (default: current directory).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/japanese_bpe.model"),
        help="Path to the trained SentencePiece model.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="IndexTTS config YAML (used to instantiate UnifiedVoice).",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        type=Path,
        default=Path("checkpoints/gpt.pth"),
        help="Base UnifiedVoice checkpoint for conditioning extraction.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ja",
        help="Language hint passed to the TextNormalizer/TextTokenizer.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device (cuda or cpu).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.01,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for split shuffling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples for debugging (0 means process all).",
    )
    parser.add_argument(
        "--audio-sr",
        type=int,
        default=24000,
        help="Target sampling rate for cached waveform if stored (kept for completeness).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose feature files already exist in output_dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to process concurrently. Increase for higher throughput if VRAM allows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of background worker threads for audio loading/resampling. 0 disables threading.",
    )
    return parser.parse_args()


SPEAKER_PATTERN = re.compile(r"^\s*(?:speaker|spk)\s*\d+\s*[:：]\s*", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = SPEAKER_PATTERN.sub("", text)
    return text.strip()


def load_audio(path: Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


class SemanticExtractor:
    def __init__(self, stats_path: Path, device: torch.device):
        self.device = device
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=stats_path
        )
        self.semantic_model = self.semantic_model.to(device)
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        self.semantic_model.eval()

    @torch.inference_mode()
    def extract(
        self,
        waveforms: Sequence[torch.Tensor] | torch.Tensor,
        sample_rates: Sequence[int] | int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(waveforms, torch.Tensor):
            waveforms = [waveforms]
        if isinstance(sample_rates, int):
            sample_rates = [sample_rates]

        arrays: List[np.ndarray] = []
        for wav, sr in zip(waveforms, sample_rates):
            current = wav
            if sr != 16000:
                current = torchaudio.functional.resample(current, sr, 16000)
            arrays.append(current.squeeze(0).cpu().numpy())

        inputs = self.feature_extractor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = outputs.hidden_states[17]
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat, attention_mask


def build_unified_voice(cfg, checkpoint: Path, device: torch.device) -> UnifiedVoice:
    gpt = UnifiedVoice(**cfg.gpt)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    gpt.load_state_dict(state, strict=False)
    gpt = gpt.to(device)
    gpt.eval()
    return gpt


def ensure_dirs(root: Path) -> Dict[str, Path]:
    subdirs = {
        "codes": root / "codes",
        "condition": root / "condition",
        "emo": root / "emo_vec",
        "text": root / "text_ids",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def resolve_audio_path(audio_value: str, audio_roots: Iterable[Path]) -> Path | None:
    path = Path(audio_value).expanduser()
    if path.is_file():
        return path

    audio_rel = Path(audio_value)
    for root in audio_roots:
        candidate = (root / audio_rel).expanduser()
        if candidate.is_file():
            return candidate
    return None


def format_audio_reference(audio_value: str, resolved_path: Path, audio_roots: Iterable[Path]) -> str:
    value = (audio_value or "").strip()
    if value:
        # Convert backslashes to forward slashes and drop any leading './'
        cleaned = Path(value).as_posix().lstrip("./")
        if cleaned:
            return cleaned

    resolved_path = resolved_path.resolve()
    for root in audio_roots:
        try:
            rel = resolved_path.relative_to(root.resolve())
            cleaned = rel.as_posix().lstrip("./")
            if cleaned:
                return cleaned
        except ValueError:
            continue
    # Fall back to filename only as a last resort
    return resolved_path.name


def process_batch(
    samples: Sequence[Dict[str, Any]],
    tokenizer: TextTokenizer,
    semantic_codec,
    semantic_extractor: SemanticExtractor,
    gpt: UnifiedVoice,
    dirs: Dict[str, Path],
    audio_roots: Iterable[Path],
    executor: ThreadPoolExecutor | None,
) -> Tuple[List[Dict[str, Any]], int]:
    prepared: List[Dict[str, Any]] = []
    skipped = 0

    candidates: List[Dict[str, Any]] = []
    for sample in samples:
        audio_field = sample.get("audio", "")
        audio_path = resolve_audio_path(audio_field, audio_roots)
        if audio_path is None:
            skipped += 1
            continue
        audio_reference = format_audio_reference(audio_field, audio_path, audio_roots)

        text = clean_text(sample.get("text", ""))
        text_tokens = tokenizer.tokenize(text)
        if not text_tokens:
            skipped += 1
            continue
        text_ids = np.asarray(tokenizer.convert_tokens_to_ids(text_tokens), dtype=np.int32)

        candidates.append(
            {
                "sample": sample,
                "audio_reference": audio_reference,
                "text": text,
                "text_ids": text_ids,
                "audio_path": audio_path,
            }
        )

    if not candidates:
        return [], skipped

    if executor is not None:
        futures: Dict[Future, Dict[str, Any]] = {}
        for item in candidates:
            future = executor.submit(load_audio, item["audio_path"], 24000)
            futures[future] = item
        for future, item in futures.items():
            try:
                waveform, sr = future.result()
            except Exception:
                skipped += 1
                continue
            item["waveform"] = waveform
            item["sr"] = sr
            prepared.append(item)
    else:
        for item in candidates:
            try:
                waveform, sr = load_audio(item["audio_path"], target_sr=24000)
            except Exception:
                skipped += 1
                continue
            item["waveform"] = waveform
            item["sr"] = sr
            prepared.append(item)

    if not prepared:
        return [], skipped

    waveforms = [item["waveform"] for item in prepared]
    sample_rates = [item["sr"] for item in prepared]
    feat, attention_mask = semantic_extractor.extract(waveforms, sample_rates)

    with torch.inference_mode():
        semantic_code, _ = semantic_codec.quantize(feat)
        if semantic_code.dim() == 1:
            semantic_code = semantic_code.unsqueeze(0)
        semantic_code = semantic_code.detach().cpu().numpy().astype(np.int32)
        cond_lengths = attention_mask.sum(dim=1).long()
        feat_t = feat.transpose(1, 2)
        cond_lengths_device = cond_lengths.to(feat.device)
        conditioning = gpt.get_conditioning(feat_t, cond_lengths_device)
        emo_vec = gpt.get_emovec(feat, cond_lengths_device)

    conditioning_np = conditioning.detach().cpu().numpy().astype(np.float32)
    emo_vec_np = emo_vec.detach().cpu().numpy().astype(np.float32)

    entries: List[Dict[str, Any]] = []
    output_root = dirs["codes"].parent
    for idx, item in enumerate(prepared):
        sample = item["sample"]
        uid = sample["id"]
        code_path = dirs["codes"] / f"{uid}.npy"
        cond_path = dirs["condition"] / f"{uid}.npy"
        emo_path = dirs["emo"] / f"{uid}.npy"
        text_path = dirs["text"] / f"{uid}.npy"

        save_numpy(code_path, semantic_code[idx])
        save_numpy(cond_path, conditioning_np[idx])
        save_numpy(emo_path, emo_vec_np[idx])
        save_numpy(text_path, item["text_ids"])

        entry = {
            "id": uid,
            "audio_path": item["audio_reference"],
            "text": item["text"],
            "speaker": sample.get("speaker", ""),
            "language": sample.get("language", ""),
            "duration": sample.get("duration"),
            "text_ids_path": text_path.relative_to(output_root).as_posix(),
            "text_len": int(item["text_ids"].size),
            "codes_path": code_path.relative_to(output_root).as_posix(),
            "code_len": int(semantic_code[idx].size),
            "condition_path": cond_path.relative_to(output_root).as_posix(),
            "condition_len": int(conditioning_np[idx].shape[0]),
            "emo_vec_path": emo_path.relative_to(output_root).as_posix(),
        }
        entries.append(entry)

    return entries, skipped


LANGUAGE_HINT_OVERRIDES: Dict[str, Optional[str]] = {
    "en": "en",
    "fr": "en",
    "de": "en",
    "zh": "zh",
    "cn": "zh",
    "ja": "ja",
    "jp": "ja",
}


def language_hint_from_code(code: str, default: Optional[str] = None) -> Optional[str]:
    return LANGUAGE_HINT_OVERRIDES.get(code.lower(), default)


def parse_dataset_spec(spec: str, output_root: Optional[Path]) -> tuple[str, Path, Path]:
    parts = spec.split("=")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            f"Invalid --dataset entry '{spec}'. Expected format LANG=MANIFEST or LANG=MANIFEST=OUTPUT."
        )
    lang = parts[0].strip()
    manifest = Path(parts[1].strip())
    if len(parts) == 3 and parts[2].strip():
        output_dir = Path(parts[2].strip())
    else:
        if output_root is not None:
            output_dir = output_root / f"{lang.lower()}_processed_data"
        else:
            output_dir = Path(f"{lang.lower()}_processed_data")
    return lang, manifest, output_dir


def preprocess_dataset(
    manifest_path: Path,
    output_dir: Path,
    dataset_language: str,
    normalizer_hint: Optional[str],
    tokenizer_path: Path,
    cfg,
    device: torch.device,
    semantic_extractor: SemanticExtractor,
    semantic_codec,
    gpt: UnifiedVoice,
    args,
    batch_size: int,
    executor: Optional[ThreadPoolExecutor],
) -> tuple[int, int, int, int]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(output_dir)

    tokenizer = TextTokenizer(
        str(tokenizer_path),
        TextNormalizer(preferred_language=normalizer_hint),
    )

    train_manifest_path = output_dir / "train_manifest.jsonl"
    val_manifest_path = output_dir / "val_manifest.jsonl"
    stats_output_path = output_dir / "stats.json"

    train_ids = load_existing_ids(train_manifest_path)
    val_ids = load_existing_ids(val_manifest_path)

    train_file = open(train_manifest_path, "a", encoding="utf-8")
    val_file = open(val_manifest_path, "a", encoding="utf-8")

    processed = 0
    skipped = 0
    pending: List[Dict[str, Any]] = []
    audio_roots = list(
        dict.fromkeys(
            [
                Path(".").resolve(),
                manifest_path.parent.resolve(),
                manifest_path.parent.parent.resolve(),
            ]
        )
    )

    def flush(force: bool = False) -> None:
        nonlocal pending, processed, skipped
        while pending and (
            force
            or len(pending) >= batch_size
            or (args.max_samples and processed + len(pending) >= args.max_samples)
        ):
            limit = batch_size
            if args.max_samples:
                remaining = args.max_samples - processed
                if remaining <= 0:
                    pending.clear()
                    return
                limit = min(limit, remaining)
            batch_records = pending[:limit]
            entries, batch_skipped = process_batch(
                batch_records,
                tokenizer,
                semantic_codec,
                semantic_extractor,
                gpt,
                dirs,
                audio_roots=audio_roots,
                executor=executor,
            )
            skipped += batch_skipped
            pending = pending[limit:]
            for entry in entries:
                is_val = assign_to_validation(entry["id"], args.val_ratio)
                if is_val:
                    if entry["id"] not in val_ids:
                        val_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        val_ids.add(entry["id"])
                else:
                    if entry["id"] not in train_ids:
                        train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        train_ids.add(entry["id"])
                processed += 1
                if args.max_samples and processed >= args.max_samples:
                    pending.clear()
                    return

    try:
        with open(manifest_path, "r", encoding="utf-8") as source:
            for line in tqdm(source, desc=f"Preprocessing [{dataset_language}]", unit="sample"):
                if args.max_samples and processed >= args.max_samples:
                    break
                record = json.loads(line)
                uid = record["id"]
                if args.skip_existing and (
                    (output_dir / "codes" / f"{uid}.npy").exists()
                    and (output_dir / "text_ids" / f"{uid}.npy").exists()
                ):
                    continue

                pending.append(record)
                flush()
                if args.max_samples and processed >= args.max_samples:
                    break
        flush(force=True)
    finally:
        train_file.close()
        val_file.close()

    update_stats_file(
        stats_output_path,
        train_ids,
        val_ids,
        tokenizer_path,
        args.gpt_checkpoint,
    )

    print(
        f"[{dataset_language}] processed={processed} skipped={skipped} "
        f"train={len(train_ids)} val={len(val_ids)} -> {output_dir}"
    )

    return processed, skipped, len(train_ids), len(val_ids)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch_size = max(1, args.batch_size)

    output_root = args.output_root.expanduser().resolve() if args.output_root else None

    dataset_specs: List[tuple[str, Path, Path]] = []
    if args.dataset:
        for spec in args.dataset:
            lang, manifest, output_dir = parse_dataset_spec(spec, output_root)
            dataset_specs.append((lang, manifest, output_dir))
    else:
        dataset_specs.append((args.language, args.manifest, args.output_dir))

    executor: Optional[ThreadPoolExecutor] = None
    if args.workers > 0:
        executor = ThreadPoolExecutor(max_workers=args.workers)

    cfg = OmegaConf.load(args.config)
    stats_value = OmegaConf.select(cfg, "w2v_stat")
    stats_path = Path(stats_value or "checkpoints/wav2vec2bert_stats.pt")
    if not stats_path.is_absolute():
        stats_path = (args.config.parent / stats_path).resolve()
    semantic_extractor = SemanticExtractor(stats_path, device)

    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    )
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()

    gpt = build_unified_voice(cfg, args.gpt_checkpoint, device)

    summaries: List[tuple[str, int, int, int, int]] = []
    try:
        for lang, manifest, output_dir in dataset_specs:
            hint_default = args.language if not args.dataset else None
            normalizer_hint = language_hint_from_code(lang, hint_default)
            processed, skipped, train_count, val_count = preprocess_dataset(
                manifest,
                output_dir,
                lang,
                normalizer_hint,
                args.tokenizer,
                cfg,
                device,
                semantic_extractor,
                semantic_codec,
                gpt,
                args,
                batch_size,
                executor,
            )
            summaries.append((lang, processed, skipped, train_count, val_count))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if len(summaries) > 1:
        print("=== Summary ===")
        for lang, processed, skipped, train_count, val_count in summaries:
            print(
                f"[{lang}] processed={processed} skipped={skipped} "
                f"train={train_count} val={val_count}"
            )


if __name__ == "__main__":
    main()
