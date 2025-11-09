#!/usr/bin/env python3
"""
Regenerate text token IDs for an existing manifest without reprocessing audio features.

Usage example:
    uv run python tools/process_text_ids.py \
        --manifest en_processed_data/train_manifest.jsonl \
        --tokenizer checkpoints/en_bpe.model \
        --output-dir en_processed_data_text_ids \
        --language-field language \
        --language en

This script reads the manifest, optionally applies language-specific romanization
(currently implemented for Japanese via pykakasi), re-tokenizes it with the existing
SentencePiece tokenizer, and writes the resulting numpy arrays to <output-dir>/text_ids.
It also emits a new manifest JSONL with updated `text_ids_path` and `text_len`
fields (and, if requested, the normalized text string).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional

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
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece tokenizer model to use for encoding.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for new text_ids and manifest.")
    parser.add_argument("--output-manifest", type=Path, default=None, help="Optional output manifest path (defaults to <output_dir>/manifest.jsonl).")
    parser.add_argument("--romanize", action="store_true", help="Apply language-specific romanization before tokenization.")
    parser.add_argument("--romanize-languages", type=str, default="ja", help="Comma-separated ISO language codes eligible for --romanize (default: ja). Provide an empty string to romanize every supported language.")
    parser.add_argument("--update-text", action="store_true", help="Write the normalized/romanized text back into the manifest.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries whose text_ids file already exists in the output directory.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of samples to process (0 = all).")
    parser.add_argument("--language", type=str, default=None, help="Fallback language code used when entries are missing language metadata.")
    parser.add_argument("--language-field", type=str, default="language", help="Manifest field containing the language code; set to '' to disable per-entry overrides.")
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


ROMANIZER_MAP: dict[str, Callable[[str], str]] = {
    "ja": to_hiragana,
}


def romanize_text(text: str, language: Optional[str]) -> str:
    normalized_lang = normalize_language_code(language)
    if normalized_lang is None:
        raise ValueError(
            "Unable to romanize text because no language code was provided. "
            "Supply --language or include a language field in the manifest."
        )
    romanizer = ROMANIZER_MAP.get(normalized_lang)
    if romanizer is None:
        supported = ", ".join(sorted(ROMANIZER_MAP))
        raise ValueError(f"Romanization not implemented for language '{normalized_lang}'. Supported languages: {supported}")
    return romanizer(text)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def normalize_language_code(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip().lower()
    return normalized or None


def parse_language_filter(spec: Optional[str]) -> Optional[set[str]]:
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None
    languages = {normalize_language_code(part) for part in spec.split(",")}
    languages.discard(None)
    return languages or None


def resolve_sample_language(record: dict[str, Any], language_field: Optional[str], fallback_language: Optional[str]) -> Optional[str]:
    if language_field:
        value = record.get(language_field)
        normalized = normalize_language_code(value)
        if normalized:
            return normalized
    return fallback_language


def process_manifest(args: argparse.Namespace) -> None:
    output_root = args.output_dir.resolve()
    text_dir = output_root / "text_ids"
    text_dir.mkdir(parents=True, exist_ok=True)

    output_manifest = args.output_manifest
    if output_manifest is None:
        output_manifest = output_root / args.manifest.name
    output_manifest = output_manifest.resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    fallback_language = normalize_language_code(args.language)
    language_field = args.language_field.strip() if args.language_field else None
    if language_field == "":
        language_field = None
    romanize_languages = parse_language_filter(args.romanize_languages) if args.romanize else None
    single_romanize_language = None
    if romanize_languages and len(romanize_languages) == 1:
        single_romanize_language = next(iter(romanize_languages))

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
            sample_language = resolve_sample_language(record, language_field, fallback_language)
            tokenizer_language = sample_language or fallback_language
            should_romanize = args.romanize
            if should_romanize and romanize_languages is not None:
                if tokenizer_language is None:
                    should_romanize = single_romanize_language is not None
                else:
                    should_romanize = tokenizer_language in romanize_languages
            if should_romanize:
                language_for_romanization = tokenizer_language
                if language_for_romanization is None and single_romanize_language is not None:
                    language_for_romanization = single_romanize_language
                text = romanize_text(text, language_for_romanization)

            text_ids = tokenizer.encode(text, language=tokenizer_language, out_type=int)
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
