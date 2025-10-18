#!/usr/bin/env python3
"""
End-to-end finetuning entry point for IndexTTS2 (GPT module) with Japanese data.

This trainer expects the preprocessing pipeline to have produced manifests where each
sample record stores paths to:
  - text token ids (.npy, int32)
  - semantic codes (.npy, int32)
  - conditioning latent (.npy, float32 [32, hidden])
  - emotion vector (.npy, float32 [hidden])

The model is optimised with cross-entropy losses over text tokens and semantic codes,
with optional gradient accumulation and mixed-precision support. Checkpoints are
emitted every 1k optimiser steps (`model_step{N}.pth`), keeping only the three most
recent snapshots. TensorBoard summaries track losses and learning rate under the
chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.front import TextNormalizer, TextTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 GPT on Japanese data.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Training manifest JSONL.")
    parser.add_argument("--val-manifest", type=Path, required=True, help="Validation manifest JSONL.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model path.")
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size per optimisation step.")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=0, help="Validation frequency in steps (0 = once per epoch).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--text-loss-weight", type=float, default=0.2, help="Weight for text CE loss.")
    parser.add_argument("--mel-loss-weight", type=float, default=0.8, help="Weight for semantic CE loss.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


@dataclass
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int


class JapaneseGPTDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.samples: List[Sample] = []
        skipped = 0
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample = Sample(
                    id=record["id"],
                    text_ids_path=Path(record["text_ids_path"]),
                    codes_path=Path(record["codes_path"]),
                    condition_path=Path(record["condition_path"]),
                    emo_vec_path=Path(record["emo_vec_path"]),
                    text_len=int(record["text_len"]),
                    code_len=int(record["code_len"]),
                    condition_len=int(record.get("condition_len", 32)),
                )
                if not (
                    sample.text_ids_path.exists()
                    and sample.codes_path.exists()
                    and sample.condition_path.exists()
                    and sample.emo_vec_path.exists()
                ):
                    skipped += 1
                    continue
                self.samples.append(sample)

        if not self.samples:
            raise RuntimeError(f"No entries found in manifest: {manifest_path}")
        if skipped:
            print(f"[Info] Skipped {skipped} entries in {manifest_path} due to missing feature files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text_ids = np.load(sample.text_ids_path, allow_pickle=False).astype(np.int64)
        codes = np.load(sample.codes_path, allow_pickle=False).astype(np.int64)
        condition = np.load(sample.condition_path, allow_pickle=False).astype(np.float32)
        emo_vec = np.load(sample.emo_vec_path, allow_pickle=False).astype(np.float32)

        return {
            "id": sample.id,
            "text_ids": torch.from_numpy(text_ids),
            "codes": torch.from_numpy(codes),
            "condition": torch.from_numpy(condition),  # [cond_len, dim]
            "emo_vec": torch.from_numpy(emo_vec),
            "text_len": torch.tensor(sample.text_len, dtype=torch.long),
            "code_len": torch.tensor(sample.code_len, dtype=torch.long),
            "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.stack(condition_tensors, dim=0)
    emo_stacked = torch.stack(emo_tensors, dim=0)

    text_lengths = torch.stack([item["text_len"] for item in batch])
    code_lengths = torch.stack([item["code_len"] for item in batch])
    cond_lengths = torch.stack([item["condition_len"] for item in batch])

    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
    }


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer(preferred_language="ja")
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(cfg_path: Path, tokenizer: TextTokenizer, base_checkpoint: Path, device: torch.device) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size

    model = UnifiedVoice(**cfg.gpt)
    checkpoint = torch.load(base_checkpoint, map_location="cpu")
    raw_state_dict = checkpoint.get("model", checkpoint)

    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state_dict[new_key] = value
    state_dict = filtered_state_dict

    resizable_keys = {
        "text_embedding.weight": model.text_embedding.weight,
        "text_head.weight": model.text_head.weight,
        "text_head.bias": model.text_head.bias,
    }
    for key, param in resizable_keys.items():
        weight = state_dict.pop(key, None)
        if weight is None:
            continue
        with torch.no_grad():
            slices = tuple(min(a, b) for a, b in zip(param.shape, weight.shape))
            if param.ndim == 1:
                param[: slices[0]].copy_(weight[: slices[0]])
            else:
                param[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])
        state_dict[key] = param.detach().clone()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys during load: {unexpected}")

    return model.to(device)


def compute_losses(
    model: UnifiedVoice,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)

    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )

    duration_zero = model.speed_emb(torch.zeros_like(use_speed))
    duration_one = model.speed_emb(torch.ones_like(use_speed))
    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1), duration_one.unsqueeze(1), duration_zero.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )

    text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    metrics = {}
    with torch.no_grad():
        mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_targets_flat = mel_targets.reshape(-1)
        mel_mask_flat = mel_mask.reshape(-1)
        if mel_mask_flat.any():
            valid_logits = mel_logits_flat[mel_mask_flat]
            valid_targets = mel_targets_flat[mel_mask_flat]
            top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
        else:
            top1 = 0.0
        metrics["mel_top1"] = top1

    return text_loss, mel_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    recent_checkpoints: List[str],
    extra: Dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "recent_checkpoints": recent_checkpoints,
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def evaluate(model: UnifiedVoice, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            text_loss, mel_loss, metrics = compute_losses(model, batch, device)
            bsz = batch["text_ids"].size(0)
            totals["text_loss"] += text_loss.item() * bsz
            totals["mel_loss"] += mel_loss.item() * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
    model.train()
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))

    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(args.config, tokenizer, args.base_checkpoint, device)

    train_dataset = JapaneseGPTDataset(args.train_manifest)
    val_dataset = JapaneseGPTDataset(args.val_manifest)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(train_loader)) // max(1, args.grad_accumulation)
    total_steps = max(total_steps, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    global_step = 0
    start_epoch = 0
    recent_checkpoints: List[str] = []
    last_saved_step: int | None = None

    resume_path: str | None = None
    if args.resume:
        if args.resume == "auto":
            candidate = output_dir / "latest.pth"
            if candidate.exists():
                resume_path = str(candidate)
        else:
            resume_path = args.resume
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)
        recent_checkpoints = checkpoint.get("recent_checkpoints", [])
        last_saved_step = checkpoint.get("step")
        print(f"[Info] Resumed from {resume_path} at epoch {start_epoch}, step {global_step}.")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    save_every = 1000
    best_val = math.inf

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                text_loss, mel_loss, metrics = compute_losses(model, batch, device)
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
            if use_amp:
                scaler.scale(loss / args.grad_accumulation).backward()
            else:
                (loss / args.grad_accumulation).backward()

            if (batch_idx + 1) % args.grad_accumulation == 0:
                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_interval == 0:
                    writer.add_scalar("train/text_loss", text_loss.item(), global_step)
                    writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    print(
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"text_loss={text_loss.item():.4f} mel_loss={mel_loss.item():.4f} "
                        f"mel_top1={metrics['mel_top1']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                    )

                if global_step % save_every == 0:
                    ckpt_path = output_dir / f"model_step{global_step}.pth"
                    recent_checkpoints.append(str(ckpt_path))
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        recent_checkpoints,
                        extra={"type": "step"},
                    )
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict() if scaler else None,
                            "epoch": epoch,
                            "step": global_step,
                            "recent_checkpoints": recent_checkpoints,
                        },
                        output_dir / "latest.pth",
                    )
                    while len(recent_checkpoints) > 3:
                        obsolete = recent_checkpoints.pop(0)
                        try:
                            os.remove(obsolete)
                        except OSError:
                            pass
                    last_saved_step = global_step

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.val_interval == 0 or global_step % args.val_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
            writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
            writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
            print(
                f"[Val] epoch={epoch + 1} step={global_step} "
                f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                f"mel_top1={val_metrics['mel_top1']:.4f}"
            )
            if val_metrics["mel_loss"] < best_val:
                best_val = val_metrics["mel_loss"]


    if global_step > 0 and last_saved_step != global_step:
        ckpt_path = output_dir / f"model_step{global_step}.pth"
        recent_checkpoints.append(str(ckpt_path))
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            recent_checkpoints,
            extra={"type": "step-final"},
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "step": global_step,
                "recent_checkpoints": recent_checkpoints,
            },
            output_dir / "latest.pth",
        )
        while len(recent_checkpoints) > 3:
            obsolete = recent_checkpoints.pop(0)
            try:
                os.remove(obsolete)
            except OSError:
                pass

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
