#!/usr/bin/env python3
"""
Utility to construct prompt/target pair manifests for GPT training without
re-running the full preprocessing pipeline.

The script consumes an existing manifest (the one produced by
`tools/preprocess_japanese.py`) and emits a new JSONL file where every line
describes a training sample formed by pairing two utterances from the same
speaker:

    - The *prompt* side contributes the conditioning latent and emotion vector.
    - The *target* side contributes the text token ids and semantic codes the
      GPT should predict.

This mirrors the pairing strategy described in the IndexTTS2 paper (different
utterances per speaker for prompt vs target) while letting us reuse the cached
feature files already on disk.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompt-target pair manifest for GPT training."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Source JSONL manifest with per-utterance features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the paired JSONL manifest.",
    )
    parser.add_argument(
        "--pairs-per-target",
        type=int,
        default=2,
        help="Number of unique prompt samples to pair with each target utterance.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional cap on the total number of pairs written (0 = no cap).",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=1,
        help="Skip targets whose text length is below this threshold.",
    )
    parser.add_argument(
        "--min-code-len",
        type=int,
        default=1,
        help="Skip targets whose semantic code length is below this threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for sampling prompt partners.",
    )
    return parser.parse_args()


@dataclass
class Sample:
    record: Dict

    @property
    def id(self) -> str:
        return self.record["id"]

    @property
    def speaker(self) -> str:
        value = self.record.get("speaker")
        if value:
            return value
        # Fallback: derive from audio filename JA_<speaker>_Wxxxxx.(ext)
        audio_path = self.record.get("audio_path", "")
        stem = Path(audio_path).stem
        match = re.match(r"(?:JA_)?(?P<spk>[^_]+)_", stem)
        if not match:
            raise ValueError(f"Unable to infer speaker from audio path '{audio_path}'")
        return match.group("spk")

    @property
    def text_len(self) -> int:
        return int(self.record.get("text_len", 0))

    @property
    def code_len(self) -> int:
        return int(self.record.get("code_len", 0))


def read_manifest(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(Sample(record))
    return samples


def group_by_speaker(samples: Iterable[Sample]) -> Dict[str, List[Sample]]:
    grouped: Dict[str, List[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.speaker, []).append(sample)
    return grouped


def build_pairs(
    grouped: Dict[str, List[Sample]],
    pairs_per_target: int,
    min_text_len: int,
    min_code_len: int,
    max_pairs: Optional[int] = None,
) -> List[Dict]:
    output: List[Dict] = []
    for speaker, items in grouped.items():
        if len(items) < 2:
            continue  # cannot form prompt/target pair

        # Shuffle once to randomise selection order.
        random.shuffle(items)

        for target in items:
            if target.text_len < min_text_len or target.code_len < min_code_len:
                continue

            prompts = [sample for sample in items if sample.id != target.id]
            if not prompts:
                continue

            random.shuffle(prompts)
            chosen = prompts[: min(pairs_per_target, len(prompts))]

            for prompt in chosen:
                pair_record = {
                    "id": f"{target.id}__{prompt.id}",
                    "speaker": speaker,
                    "prompt_id": prompt.id,
                    "prompt_audio_path": prompt.record.get("audio_path", ""),
                    "prompt_condition_path": prompt.record["condition_path"],
                    "prompt_condition_len": int(prompt.record.get("condition_len", 0)),
                    "prompt_emo_vec_path": prompt.record.get("emo_vec_path", ""),
                    "prompt_duration": prompt.record.get("duration"),
                    "target_id": target.id,
                    "target_audio_path": target.record.get("audio_path", ""),
                    "target_text": target.record.get("text", ""),
                    "target_text_ids_path": target.record["text_ids_path"],
                    "target_text_len": target.text_len,
                    "target_codes_path": target.record["codes_path"],
                    "target_code_len": target.code_len,
                    "target_emo_vec_path": target.record.get("emo_vec_path", ""),
                }
                output.append(pair_record)
                if max_pairs and len(output) >= max_pairs:
                    return output
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    samples = read_manifest(args.manifest)
    if not samples:
        raise RuntimeError(f"No entries found in {args.manifest}")

    grouped = group_by_speaker(samples)
    max_pairs = args.max_pairs if args.max_pairs > 0 else None
    pairs = build_pairs(
        grouped,
        pairs_per_target=args.pairs_per_target,
        min_text_len=args.min_text_len,
        min_code_len=args.min_code_len,
        max_pairs=max_pairs,
    )

    if not pairs:
        raise RuntimeError("No valid prompt-target pairs were generated.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in pairs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[Done] Wrote {len(pairs)} pairs covering {len(grouped)} speakers "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
