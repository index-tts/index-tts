#!/usr/bin/env python3
"""
Extend an existing SentencePiece model by appending new tokens while keeping
the original token order and indices intact.

The workflow trains a temporary tokenizer on additional corpora, harvests
candidate pieces, filters out ones already present in the base model, and
appends the remaining pieces (in rank order) to reach the requested target
vocabulary size.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import sentencepiece as spm

# Ensure we can import the legacy protobuf builder helpers used by SentencePiece.
try:  # pragma: no cover - import side-effect
    from google.protobuf.internal import builder as _builder  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - fallback shim
    from tools.tokenizer import protobuf_builder_compat as _builder  # noqa: F401
    sys.modules["google.protobuf.internal.builder"] = _builder

from sentencepiece import sentencepiece_model_pb2 as sp_model  # noqa: E402

DEFAULT_TARGET_SIZE = 24_000
DEFAULT_EXTRA_FACTOR = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append new tokens to an existing SentencePiece model.")
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Path to the existing SentencePiece model (.model).",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        nargs="+",
        required=True,
        help="JSONL manifests (each line must contain a 'text' field).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        help="Destination for the extended .model (defaults to <base>_extended.model).",
    )
    parser.add_argument(
        "--output-vocab",
        type=Path,
        help="Destination for the extended .vocab (defaults to match output-model prefix).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Desired total vocabulary size after extension (default: {DEFAULT_TARGET_SIZE}).",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        help="Temporary tokenizer vocab size used to mine new tokens (default: target-size * extra-factor).",
    )
    parser.add_argument(
        "--extra-factor",
        type=float,
        default=DEFAULT_EXTRA_FACTOR,
        help="Multiplier applied to target-size when computing candidate-size (ignored if candidate-size set).",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage for the temporary tokenizer.",
    )
    parser.add_argument(
        "--model-type",
        choices=["bpe", "unigram"],
        default="bpe",
        help="SentencePiece model type for the temporary tokenizer.",
    )
    return parser.parse_args()


def validate_inputs(base_model: Path, manifests: Iterable[Path]) -> None:
    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    missing = [str(p) for p in manifests if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing manifest(s): {', '.join(missing)}")


def collect_corpus(manifests: Iterable[Path]) -> tuple[Path, int]:
    """Write all text fields from the manifests into a temporary corpus file."""
    total = 0
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt")
    path = Path(tmp.name)
    try:
        with tmp:
            for manifest in manifests:
                with manifest.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        text = record.get("text", "")
                        text = text.strip()
                        if not text:
                            continue
                        tmp.write(text + "\n")
                        total += 1
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if total == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError("No usable text samples found in manifests.")
    return path, total


def train_candidate_tokenizer(
    corpus: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
) -> sp_model.ModelProto:
    """Train a temporary tokenizer and return its ModelProto."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "candidate"
        spm.SentencePieceTrainer.Train(
            input=str(corpus),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            train_extremely_large_corpus=True,
            input_sentence_size=0,
            shuffle_input_sentence=True,
            bos_id=0,
            eos_id=1,
            unk_id=2,
            pad_id=-1,
        )
        candidate_model_path = prefix.with_suffix(".model")
        proto = sp_model.ModelProto()
        proto.ParseFromString(candidate_model_path.read_bytes())
    return proto


def load_model(path: Path) -> sp_model.ModelProto:
    proto = sp_model.ModelProto()
    proto.ParseFromString(path.read_bytes())
    return proto


def append_tokens(
    base_proto: sp_model.ModelProto,
    candidate_proto: sp_model.ModelProto,
    target_size: int,
) -> list[sp_model.ModelProto.SentencePiece]:
    """Append new pieces from candidate_proto to base_proto to reach target_size."""
    base_count = len(base_proto.pieces)
    if base_count >= target_size:
        raise ValueError(f"Target size ({target_size}) must exceed current vocabulary ({base_count}).")

    needed = target_size - base_count
    allowed_types = {
        sp_model.ModelProto.SentencePiece.Type.NORMAL,
        sp_model.ModelProto.SentencePiece.Type.USER_DEFINED,
    }

    existing_texts = {piece.piece for piece in base_proto.pieces}
    appended: list[sp_model.ModelProto.SentencePiece] = []

    candidate_lookup = {}
    for piece in candidate_proto.pieces:
        if piece.type in allowed_types and piece.piece not in candidate_lookup:
            candidate_lookup[piece.piece] = piece

    def add_piece(source_piece: sp_model.ModelProto.SentencePiece) -> None:
        new_piece = base_proto.pieces.add()
        new_piece.piece = source_piece.piece
        new_piece.score = source_piece.score
        new_piece.type = source_piece.type
        existing_texts.add(new_piece.piece)
        appended.append(new_piece)

    # Ensure single lowercase ASCII letters and digits exist so we can encode contractions and numbers.
    for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
        if ch in existing_texts:
            continue
        piece = candidate_lookup.get(ch)
        if piece is None:
            continue
        add_piece(piece)
        if len(appended) >= needed:
            break

    if len(appended) < needed:
        for piece in candidate_proto.pieces:
            if piece.type not in allowed_types:
                continue
            if piece.piece in existing_texts:
                continue
            add_piece(piece)
            if len(appended) >= needed:
                break

    if len(appended) < needed:
        raise RuntimeError(
            f"Only found {len(appended)} new tokens (need {needed}). "
            "Increase --candidate-size or supply more training text."
        )

    base_proto.trainer_spec.vocab_size = len(base_proto.pieces)
    return appended


def write_vocab_file(proto: sp_model.ModelProto, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for piece in proto.pieces:
            handle.write(f"{piece.piece}\t{piece.score}\n")


def main() -> int:
    args = parse_args()
    validate_inputs(args.base_model, args.manifests)

    base_proto = load_model(args.base_model)
    base_size = len(base_proto.pieces)
    if base_proto.trainer_spec.vocab_size != base_size:
        print(
            f"[extend_bpe] Warning: trainer_spec.vocab_size ({base_proto.trainer_spec.vocab_size}) "
            f"differs from actual pieces ({base_size}). Using piece count.",
            file=sys.stderr,
        )

    target_size = args.target_size
    if target_size <= base_size:
        raise ValueError(
            f"Target size ({target_size}) must be greater than current vocabulary ({base_size})."
        )

    candidate_size = args.candidate_size or int(target_size * args.extra_factor)
    candidate_size = max(candidate_size, target_size + 1024)  # ensure some headroom

    corpus_path, total_samples = collect_corpus(args.manifests)
    print(f"[extend_bpe] Collected {total_samples} samples into {corpus_path}")
    try:
        print(
            f"[extend_bpe] Training candidate tokenizer (vocab_size={candidate_size}, "
            f"model_type={args.model_type})"
        )
        candidate_proto = train_candidate_tokenizer(
            corpus=corpus_path,
            vocab_size=candidate_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
        )
    finally:
        corpus_path.unlink(missing_ok=True)

    appended = append_tokens(base_proto, candidate_proto, target_size=target_size)
    print(f"[extend_bpe] Appended {len(appended)} new tokens (total={len(base_proto.pieces)})")

    output_model = args.output_model or args.base_model.with_name(args.base_model.stem + "_extended.model")
    output_vocab = args.output_vocab or output_model.with_suffix(".vocab")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_bytes(base_proto.SerializeToString())
    write_vocab_file(base_proto, output_vocab)

    print(f"[extend_bpe] Wrote extended model to {output_model}")
    print(f"[extend_bpe] Wrote extended vocab to {output_vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
