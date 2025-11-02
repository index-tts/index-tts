#!/usr/bin/env python3
"""
Regenerate text token IDs for an existing manifest without reprocessing audio features.

Usage example:
    uv run python tools/process_text_ids.py \
        --manifest processed_data_2/train_manifest.jsonl \
        --tokenizer checkpoints/japanese_bpe.model \
        --output-dir processed_data_2_text_ids_2 \
        --romanize \
        --update-text

This script reads the manifest, optionally converts the text to Hiragana using
pykakasi, re-tokenizes it with the existing SentencePiece tokenizer, and writes
the resulting numpy arrays to a new directory tree (default: <output-dir>/text_ids).
It also emits a new manifest JSONL with updated `text_ids_path` and `text_len`
fields (and, if requested, the normalized text string).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from indextts.utils.front import TextNormalizer, TextTokenizer

try:
    from pykakasi import kakasi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    kakasi = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-tokenize text entries in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest JSONL.")
    parser.add_argument("--tokenizer", type=Path, default=Path("checkpoints/japanese_bpe.model"), help="SentencePiece tokenizer model.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for new text_ids and manifest.")
    parser.add_argument("--output-manifest", type=Path, default=None, help="Optional output manifest path (defaults to <output_dir>/manifest.jsonl).")
    parser.add_argument("--romanize", action="store_true", help="Convert text to Hiragana (requires pykakasi).")
    parser.add_argument("--update-text", action="store_true", help="Write the normalized/romanized text back into the manifest.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries whose text_ids file already exists in the output directory.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of samples to process (0 = all).")
    parser.add_argument("--language", type=str, default="ja", help="Language hint passed to the tokenizer (default: ja).")
    return parser.parse_args()


_kakasi = None


def ensure_kakasi():
    global _kakasi
    if _kakasi is not None:
        return _kakasi
    if kakasi is None:
        raise RuntimeError("pykakasi is required for --romanize but is not installed. Install `pykakasi` via pip.")
    inst = kakasi()
    inst.setMode("J", "H")  # Kanji to Hiragana
    inst.setMode("K", "H")  # Katakana to Hiragana
    inst.setMode("H", "H")  # Hiragana stays Hiragana
    inst.setMode("a", "H")  # Romaji to Hiragana if present
    _kakasi = inst
    return _kakasi


def to_hiragana(text: str) -> str:
    converter = ensure_kakasi()
    result = converter.convert(text)
    return "".join(item.get("hira") or item.get("orig", "") for item in result)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def process_manifest(args: argparse.Namespace) -> None:
    output_root = args.output_dir.resolve()
    text_dir = output_root / "text_ids"
    text_dir.mkdir(parents=True, exist_ok=True)

    output_manifest = args.output_manifest
    if output_manifest is None:
        output_manifest = output_root / args.manifest.name
    output_manifest = output_manifest.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    normalizer = TextNormalizer(preferred_language=args.language)
    tokenizer = TextTokenizer(str(args.tokenizer), normalizer)

    processed = 0
    skipped = 0

    with args.manifest.open("r", encoding="utf-8") as handle, output_manifest.open("w", encoding="utf-8") as out_handle:
        for line in tqdm(handle, desc="Retokenizing", unit="utt"):
            if args.max_samples and processed >= args.max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record["id"]
            text = record.get("text", "")
            if args.romanize:
                text = to_hiragana(text)

            text_ids = tokenizer.encode(text, language=args.language, out_type=int)
            text_ids = np.asarray(text_ids, dtype=np.int32)

            text_path = text_dir / f"{sample_id}.npy"
            if args.skip_existing and text_path.exists():
                skipped += 1
                continue

            np.save(text_path, text_ids)
            processed += 1

            record["text_ids_path"] = relative_path(text_path)
            record["text_len"] = int(text_ids.size)
            if args.update_text:
                record["text"] = text

            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Processed: {processed}  Skipped: {skipped}")
    print(f"Manifest written to: {output_manifest}")


def main() -> None:
    args = parse_args()
    process_manifest(args)


if __name__ == "__main__":
    main()
