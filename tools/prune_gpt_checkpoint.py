#!/usr/bin/env python3
"""
Utility to prune training checkpoints down to the tensors required for
IndexTTS2 inference.

Typical usage:

    uv run python tools/prune_gpt_checkpoint.py \
        --input trained_ckpts_multilingual/model_step33000.pth \
        --output checkpoints/gpt_multilingual_pruned.pth

The resulting file mirrors the structure of the shipped inference
checkpoints (an OrderedDict of model weights) so it can be loaded by
`indextts/infer_v2_modded.py` and the WebUI.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip training artefacts (optimizer, scheduler, scaler, etc.) "
        "from a checkpoint and emit an inference-ready weight file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the training checkpoint (.pth) produced by train_gpt_v2.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the pruned checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        choices=("keep", "float32", "float16"),
        default="keep",
        help="Optional tensor dtype override for the saved weights (default: keep original).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report summary without writing the output file.",
    )
    return parser.parse_args()


def _coerce_dtype(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "keep":
        return tensor
    target = torch.float16 if dtype == "float16" else torch.float32
    if tensor.dtype == target:
        return tensor
    return tensor.to(dtype=target)


def summarise_state(state: OrderedDict) -> Tuple[int, int]:
    tensor_count = 0
    total_params = 0
    for value in state.values():
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            total_params += value.numel()
    return tensor_count, total_params


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input}")

    checkpoint = torch.load(args.input, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            "Unrecognised checkpoint structure. Expected either a plain state_dict "
            "or a dict containing the 'model' key produced by train_gpt_v2.py."
        )

    if not isinstance(state_dict, dict):
        raise RuntimeError("Model state should be a dict-like object.")

    dtype_choice = args.dtype

    pruned = OrderedDict()
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            tensor = _coerce_dtype(tensor, dtype_choice)
            pruned[key] = tensor
        else:
            # Non-tensor entries are extremely rare but we keep them just in case.
            pruned[key] = value

    tensor_count, param_total = summarise_state(pruned)
    print(f"[Prune] Retained {tensor_count} tensors / {param_total:,} parameters.")

    if args.dry_run:
        print("[Prune] Dry run enabled; no file was written.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"[Prune] Wrote {args.output} ({size_mb:.2f} MiB).")


if __name__ == "__main__":
    main()
