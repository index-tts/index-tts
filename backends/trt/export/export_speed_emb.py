"""Export speed_emb weight tensor from gpt.pth as a standalone file.

Usage:
    python export_speed_emb.py [--model_dir checkpoints] [--output speed_emb.pt]
"""

import argparse
import os
import torch


def export(model_dir, output_path):
    from omegaconf import OmegaConf
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    print("Loading GPT model...")
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, os.path.join(model_dir, cfg.gpt_checkpoint))
    gpt.eval()

    weight = gpt.speed_emb.weight.data.clone().cpu()
    print(f"  speed_emb.weight shape: {weight.shape}")
    print(f"  speed_emb.weight norm : {weight.norm():.6f}")

    torch.save(weight, output_path)
    print(f"Saved to {output_path} ({os.path.getsize(output_path)} bytes)")

    del gpt
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints")
    parser.add_argument("--output", default="speed_emb.pt")
    args = parser.parse_args()
    export(args.model_dir, args.output)


if __name__ == "__main__":
    main()
