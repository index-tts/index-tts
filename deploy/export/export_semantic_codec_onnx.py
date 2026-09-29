"""Export semantic_codec.quantize (encoder + VQ) to ONNX.

Usage:
    python export_semantic_codec_onnx.py [--output semantic_codec.onnx]
"""

import argparse
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import safetensors.torch
from huggingface_hub import hf_hub_download

from indextts.utils.maskgct_utils import build_semantic_codec


class SemanticCodecEncoder(nn.Module):
    """Wraps semantic_codec.quantize as a proper nn.Module for ONNX export.

    Directly calls the single FVQ codebook to avoid the quantizer dropout
    logic (which uses data-dependent comparisons incompatible with ONNX).
    """

    def __init__(self, codec):
        super().__init__()
        self.encoder = codec.encoder
        self.fvq = codec.quantizer.quantizers[0]
        self.downsample_scale = codec.downsample_scale
        if self.downsample_scale is not None and self.downsample_scale > 1:
            self.down = codec.down

    def forward(self, x):
        """
        Args:
            x: (B, T, 1024) semantic embeddings
        Returns:
            indices: (B, T) codebook indices (int64)
            quantized: (B, T, 1024) continuous quantized embeddings
        """
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = torch.nn.functional.gelu(x)
            x = x.transpose(1, 2)

        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        # Directly call the single FVQ (num_quantizers=1)
        # Returns: (quantized, commit_loss, codebook_loss, indices, perplexity)
        quantized_out, _, _, indices, _ = self.fvq(x)
        return indices, quantized_out.transpose(1, 2)


def export_onnx(output_path, device="cuda:0"):
    cfg = OmegaConf.load("checkpoints/config.yaml")
    codec = build_semantic_codec(cfg.semantic_codec)
    ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(codec, ckpt_path)
    codec = codec.to(device).eval()

    wrapper = SemanticCodecEncoder(codec).eval().to(device)

    B, T = 2, 200
    dummy = torch.randn(B, T, 1024, device=device)

    with torch.no_grad():
        indices, quantized = wrapper(dummy)
    print(f"Indices shape: {indices.shape}, Quantized shape: {quantized.shape}")

    B_dim = torch.export.Dim("B", min=1, max=8)
    T_dim = torch.export.Dim("T", min=10, max=2000)
    dynamic_shapes = {"x": {0: B_dim, 1: T_dim}}

    print("Exporting semantic_codec encoder to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            output_path,
            input_names=["x"],
            output_names=["indices", "quantized"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    print(f"ONNX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="semantic_codec.onnx")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_onnx(args.output, args.device)


if __name__ == "__main__":
    main()
