#!/usr/bin/env python3

"""
Convenience script for running IndexTTS2 inference from the command line.

Examples
--------
Generate Japanese audio with the latest fine-tuned checkpoint:

    uv run python inference_script.py \
        --config checkpoints/config_finetune.yaml \
        --speaker vivy.wav \
        --text '任務なんてクソくらえ！任務なんてクソくらえよ、ルネ！' \
        --output out.wav \
        --device cuda:0

You can either pass `--text` directly or use `--text-file` to read the input
sentence from disk. All other parameters (tokenizer path, GPT checkpoint, etc.)
are taken from the provided config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional

from indextts.infer_v2 import IndexTTS2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IndexTTS2 inference with configurable speaker prompt and text."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the YAML config that defines checkpoint/tokenizer paths (default: checkpoints/config.yaml).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Directory containing weights/tokenizer referenced in the config (default: checkpoints). "
             "If a file path is supplied, it will be treated as a GPT checkpoint override.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
        help="Reference speaker audio (wav/mp3/etc.).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        type=str,
        help="Text to synthesise.",
    )
    group.add_argument(
        "--text-file",
        type=str,
        help="Path to a UTF-8 text file containing the content to synthesise.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Destination wav file (default: output.wav).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. cuda:0, cpu). Leave unset for automatic selection.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 inference when running on CUDA.",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        type=str,
        default=None,
        help="Optional path to override the GPT checkpoint referenced in the config.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional path to override the BPE tokenizer referenced in the config.",
    )
    parser.add_argument(
        "--emo-audio",
        type=str,
        default=None,
        help="Optional emotion reference audio clip.",
    )
    parser.add_argument(
        "--emo-alpha",
        type=float,
        default=1.0,
        help="Blend factor for the emotion reference audio (default: 1.0).",
    )
    parser.add_argument(
        "--emo-text",
        type=str,
        default=None,
        help="Text to derive emotion vector from (used when --use-emo-text is supplied).",
    )
    parser.add_argument(
        "--use-emo-text",
        action="store_true",
        help="Derive the emotion vector from text using Qwen emotion model.",
    )
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=120,
        help="Maximum text tokens per segment (default: 120).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sampling top-k (leave unset to use config defaults).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Sampling top-p (leave unset to use config defaults).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (leave unset to use config defaults).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Beam search width (leave unset to use config defaults).",
    )
    parser.add_argument(
        "--interval-silence",
        type=int,
        default=200,
        help="Silence duration (ms) inserted between segments (default: 200).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose inference logs.",
    )

    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    text_path = Path(args.text_file)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    return text_path.read_text(encoding="utf-8").strip()


def build_generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.num_beams is not None:
        kwargs["num_beams"] = args.num_beams
    return kwargs


def main() -> None:
    args = parse_args()
    text = load_text(args)
    generation_kwargs = build_generation_kwargs(args)

    # Prepare configuration overrides if requested.
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    model_dir_path = Path(args.model_dir).expanduser()
    gpt_override: Optional[Path] = None

    if model_dir_path.is_file():
        gpt_override = model_dir_path
        parent = model_dir_path.parent if model_dir_path.parent != Path("") else cfg_path.parent
        model_dir_resolved = parent.resolve()
    else:
        model_dir_resolved = model_dir_path.resolve()

    if args.gpt_checkpoint:
        gpt_override = Path(args.gpt_checkpoint).expanduser()

    if gpt_override is not None:
        cfg.gpt_checkpoint = str(gpt_override.resolve())

    if args.tokenizer:
        if "dataset" not in cfg or "bpe_model" not in cfg["dataset"]:
            raise KeyError("Config does not contain dataset.bpe_model to override.")
        tokenizer_path = Path(args.tokenizer).expanduser().resolve()
        cfg.dataset["bpe_model"] = str(tokenizer_path)

    # Persist overrides to a temporary file so IndexTTS2 can read them.
    tmp_cfg_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_cfg_file:
            OmegaConf.save(cfg, tmp_cfg_file.name)
            tmp_cfg_path = Path(tmp_cfg_file.name)

        engine = IndexTTS2(
            cfg_path=str(tmp_cfg_path),
            model_dir=str(model_dir_resolved),
            device=args.device,
            use_fp16=args.fp16,
        )

        engine.infer(
            spk_audio_prompt=args.speaker,
            text=text,
            output_path=args.output,
            emo_audio_prompt=args.emo_audio,
            emo_alpha=args.emo_alpha,
            use_emo_text=args.use_emo_text,
            emo_text=args.emo_text,
            interval_silence=args.interval_silence,
            verbose=args.verbose,
            max_text_tokens_per_segment=args.max_text_tokens,
            **generation_kwargs,
        )
        print(f"Inference complete. Output saved to {Path(args.output).resolve()}")
    finally:
        if tmp_cfg_path is not None:
            try:
                tmp_cfg_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
