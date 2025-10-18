#!/usr/bin/env python3
"""
Preprocess the JA Yodas dataset for IndexTTS2 fine-tuning.

This script performs:
  1. Text cleanup + Japanese normalization and tokenisation.
  2. Audio loading and resampling.
  3. Semantic feature extraction via SeamlessM4T + Wav2Vec2Bert.
  4. Semantic code quantisation with RepCodec.
  5. Conditioning latent + emotion vector extraction with UnifiedVoice v2.
  6. Manifest generation pointing to the cached features.

Outputs are written to a root directory (default: ./processed_data) with
sub-folders for codes, conditioning latents, emotion vectors, and text ids.
Train/validation manifests are emitted as JSONL for downstream training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def update_stats_file(stats_path: Path, train_ids: set[str], val_ids: set[str], tokenizer_path: Path, checkpoint_path: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Preprocess JA Yodas dataset for IndexTTS2 fine-tuning.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("JA_yodas_dataset/ja_yodas_train.jsonl"),
        help="Source manifest (JSONL) with fields id/text/audio/speaker/language.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_data"),
        help="Directory to store processed artifacts.",
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
    return parser.parse_args()


SPEAKER_PATTERN = re.compile(r"^\s*(?:speaker|spk)\s*\d+\s*[:：]\s*", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Remove conversation markers like 'Speaker 1:' while keeping Japanese content."""
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
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            path_=stats_path
        )
        self.semantic_model = self.semantic_model.to(device)
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        self.semantic_model.eval()

    @torch.inference_mode()
    def extract(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0).cpu().numpy()
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = outputs.hidden_states[17]  # (B, T, C)
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


def process_sample(
    sample: Dict[str, Any],
    tokenizer: TextTokenizer,
    semantic_codec,
    semantic_extractor: SemanticExtractor,
    gpt: UnifiedVoice,
    dirs: Dict[str, Path],
) -> Dict[str, Any] | None:
    audio_path = Path(sample["audio"]).expanduser()
    if not audio_path.is_file():
        return None

    text = clean_text(sample.get("text", ""))
    text_tokens = tokenizer.tokenize(text, language="ja")
    if not text_tokens:
        return None
    text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    text_ids = np.asarray(text_ids, dtype=np.int32)

    waveform, sr = load_audio(audio_path, target_sr=24000)
    feat, attention_mask = semantic_extractor.extract(waveform, sr)

    with torch.inference_mode():
        semantic_code, _ = semantic_codec.quantize(feat)
        if semantic_code.dim() > 1:
            semantic_code = semantic_code.squeeze(0)
        semantic_code = semantic_code.cpu().numpy().astype(np.int32)
        cond_lengths = attention_mask.sum(dim=1).long()
        conditioning = gpt.get_conditioning(
            feat.transpose(1, 2), cond_lengths.to(feat.device)
        )
        conditioning = conditioning.cpu().numpy().astype(np.float32).squeeze(0)
        emo_vec = gpt.get_emovec(feat, cond_lengths.to(feat.device))
        emo_vec = emo_vec.cpu().numpy().astype(np.float32).squeeze(0)

    uid = sample["id"]
    code_path = dirs["codes"] / f"{uid}.npy"
    cond_path = dirs["condition"] / f"{uid}.npy"
    emo_path = dirs["emo"] / f"{uid}.npy"
    text_path = dirs["text"] / f"{uid}.npy"

    save_numpy(code_path, semantic_code)
    save_numpy(cond_path, conditioning)
    save_numpy(emo_path, emo_vec)
    save_numpy(text_path, text_ids)

    entry = {
        "id": uid,
        "audio_path": str(audio_path),
        "text": text,
        "speaker": sample.get("speaker", ""),
        "language": sample.get("language", "ja"),
        "duration": sample.get("duration"),
        "text_ids_path": str(text_path),
        "text_len": int(text_ids.size),
        "codes_path": str(code_path),
        "code_len": int(semantic_code.size),
        "condition_path": str(cond_path),
        "condition_len": int(conditioning.shape[0]),
        "emo_vec_path": str(emo_path),
    }
    return entry


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(output_dir)

    cfg = OmegaConf.load(args.config)
    tokenizer = TextTokenizer(str(args.tokenizer), TextNormalizer(preferred_language="ja"))

    stats_value = OmegaConf.select(cfg, "w2v_stat")
    stats_path = Path(stats_value or "checkpoints/wav2vec2bert_stats.pt")
    if not stats_path.is_absolute():
        stats_path = (args.config.parent / stats_path).resolve()
    semantic_extractor = SemanticExtractor(stats_path, device)

    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device)
    semantic_codec.eval()

    gpt = build_unified_voice(cfg, args.gpt_checkpoint, device)

    manifest_path = args.manifest.expanduser().resolve()
    train_manifest_path = output_dir / "train_manifest.jsonl"
    val_manifest_path = output_dir / "val_manifest.jsonl"
    stats_output_path = output_dir / "stats.json"

    train_ids = load_existing_ids(train_manifest_path)
    val_ids = load_existing_ids(val_manifest_path)

    train_file = open(train_manifest_path, "a", encoding="utf-8")
    val_file = open(val_manifest_path, "a", encoding="utf-8")

    processed = 0
    skipped = 0

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(tqdm(handle, desc="Preprocessing", unit="utt")):
                if args.max_samples and processed >= args.max_samples:
                    break
                if not line.strip():
                    continue
                payload = json.loads(line)
                sample_id = payload["id"]

                if sample_id in train_ids or sample_id in val_ids:
                    skipped += 1
                    continue

                if args.skip_existing:
                    code_path = dirs["codes"] / f"{sample_id}.npy"
                    cond_path = dirs["condition"] / f"{sample_id}.npy"
                    emo_path = dirs["emo"] / f"{sample_id}.npy"
                    text_path = dirs["text"] / f"{sample_id}.npy"
                    if all(path.exists() for path in (code_path, cond_path, emo_path, text_path)):
                        skipped += 1
                        continue

                entry = process_sample(
                    payload,
                    tokenizer,
                    semantic_codec,
                    semantic_extractor,
                    gpt,
                    dirs,
                )
                if entry is None:
                    continue

                is_val = assign_to_validation(sample_id, args.val_ratio)
                if is_val:
                    val_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    val_file.flush()
                    val_ids.add(sample_id)
                else:
                    train_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    train_file.flush()
                    train_ids.add(sample_id)

                processed += 1
                update_stats_file(stats_output_path, train_ids, val_ids, args.tokenizer, args.gpt_checkpoint)

    finally:
        train_file.close()
        val_file.close()

    total_processed = len(train_ids) + len(val_ids)
    print(f"Preprocessing complete. Processed entries: {processed}, skipped existing: {skipped}.")
    print(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)} (total tracked: {total_processed}).")


if __name__ == "__main__":
    main()
