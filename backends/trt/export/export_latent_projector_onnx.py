"""Export latent projector (merged vq2emb + gpt_layer) to ONNX.

Usage:
    python export_latent_projector_onnx.py [--output latent_projector.onnx]
"""

import argparse
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import safetensors.torch
from huggingface_hub import hf_hub_download

from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2


class LatentProjector(nn.Module):
    """Merged vq2emb + gpt_layer: returns gpt_layer(latent) + vq2emb(codes)."""

    def __init__(self, gpt_layer, quantizer):
        super().__init__()
        self.gpt_layer = gpt_layer
        self.quantizer = quantizer

    def forward(self, latent, codes):
        """
        Args:
            latent: (B, T, 1280) GPT hidden states
            codes: (B, T) codebook indices (int64)
        Returns:
            S_infer: (B, T, 1024) = gpt_layer(latent) + vq2emb(codes).transpose(1,2)
        """
        projected = self.gpt_layer(latent)
        emb = self.quantizer.vq2emb(codes.unsqueeze(0))
        emb = emb.transpose(1, 2)
        return projected + emb


def remove_weight_norm_recursive(model):
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
        except ValueError:
            pass


def export_onnx(output_path, device="cuda:0"):
    cfg = OmegaConf.load("checkpoints/config.yaml")

    codec = build_semantic_codec(cfg.semantic_codec)
    ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(codec, ckpt_path)
    codec = codec.to(device).eval()

    s2mel_path = f"checkpoints/{cfg.s2mel_checkpoint}"
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path,
                                       load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel = s2mel.to(device).eval()

    wrapper = LatentProjector(s2mel.models['gpt_layer'], codec.quantizer).eval().to(device)
    remove_weight_norm_recursive(wrapper)

    B, T = 2, 200
    dummy_latent = torch.randn(B, T, 1280, device=device)
    dummy_codes = torch.randint(0, 8192, (B, T), device=device, dtype=torch.long)

    with torch.no_grad():
        out = wrapper(dummy_latent, dummy_codes)
    print(f"Output shape: {out.shape}")

    B_dim = torch.export.Dim("B", min=1, max=8)
    T_dim = torch.export.Dim("T", min=10, max=2000)
    dynamic_shapes = {
        "latent": {0: B_dim, 1: T_dim},
        "codes": {0: B_dim, 1: T_dim},
    }

    print("Exporting latent projector to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_latent, dummy_codes),
            output_path,
            input_names=["latent", "codes"],
            output_names=["S_infer"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    print(f"ONNX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="latent_projector.onnx")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_onnx(args.output, args.device)


if __name__ == "__main__":
    main()
