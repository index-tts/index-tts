#!/usr/bin/env python3
"""
Utility script to train a SentencePiece BPE tokenizer on the Japanese corpus.

Example:
    python tools/tokenizer/train_bpe.py \\
        --manifest JA_yodas_dataset/ja_yodas_train.jsonl \\
        --output-prefix checkpoints/japanese_bpe \\
        --vocab-size 12000
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

from indextts.utils.front import TextNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Japanese BPE tokenizer with SentencePiece.")
    parser.add_argument(
        "--manifest",
        nargs="+",
        required=True,
        help="One or more JSONL manifests containing a 'text' field.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("checkpoints/japanese_bpe"),
        help="Output prefix for the tokenizer files (.model/.vocab).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=12000,
        help="Desired vocabulary size.",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage for SentencePiece (keep near 1.0 for Japanese).",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="SentencePiece model type.",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Limit the number of sentences sampled for training (0 means use all).",
    )
    parser.add_argument(
        "--byte-fallback",
        action="store_true",
        help="Enable byte fallback to avoid <unk> for unseen characters.",
    )
    return parser.parse_args()


def iter_texts(manifests: list[Path]) -> tuple[int, int, Path]:
    normalizer = TextNormalizer(preferred_language="ja")
    normalizer.load()

    num_samples = 0
    num_empty = 0
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    try:
        with tmp_file as fp:
            for manifest in manifests:
                with open(manifest, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        text = payload.get("text", "")
                        text = normalizer.normalize(text, language="ja")
                        if not text:
                            num_empty += 1
                            continue
                        fp.write(text + "\n")
                        num_samples += 1
    except Exception:
        os.unlink(tmp_file.name)
        raise
    return num_samples, num_empty, Path(tmp_file.name)


def train_tokenizer(args: argparse.Namespace) -> None:
    manifests = [Path(m).expanduser().resolve() for m in args.manifest]
    missing = [str(p) for p in manifests if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing manifest(s): {', '.join(missing)}")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    num_samples, num_empty, corpus_path = iter_texts(manifests)
    if num_samples == 0:
        raise RuntimeError("No non-empty samples found. Cannot train tokenizer.")

    spm_kwargs = {
        "input": str(corpus_path),
        "model_prefix": str(output_prefix),
        "vocab_size": args.vocab_size,
        "character_coverage": args.character_coverage,
        "model_type": args.model_type,
        "bos_id": 0,
        "eos_id": 1,
        "unk_id": 2,
        "pad_id": -1,
        "input_sentence_size": args.input_sentence_size,
        "shuffle_input_sentence": True,
        "byte_fallback": args.byte_fallback,
        "train_extremely_large_corpus": True,
    }

    print(f"[Tokenizer] Training on {num_samples} samples (skipped {num_empty}).")
    try:
        spm.SentencePieceTrainer.train(**spm_kwargs)
    finally:
        corpus_path.unlink(missing_ok=True)

    model_path = output_prefix.with_suffix(".model")
    vocab_path = output_prefix.with_suffix(".vocab")

    print(f"[Tokenizer] Saved SentencePiece model to: {model_path}")
    print(f"[Tokenizer] Saved vocabulary to: {vocab_path}")


def main() -> int:
    args = parse_args()
    try:
        train_tokenizer(args)
    except KeyboardInterrupt:
        print("Interrupted!", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
