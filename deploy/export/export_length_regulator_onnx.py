"""Export length_regulator to ONNX.

Usage:
    python export_length_regulator_onnx.py [--output length_regulator.onnx]
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2


class LengthRegulatorWrapper(nn.Module):
    """Wrapper for InterpolateRegulator that takes T_out as explicit input
    instead of computing it from ylens.max()."""

    def __init__(self, regulator):
        super().__init__()
        self.content_in_proj = regulator.content_in_proj
        self.model = regulator.model
        self.interpolate = regulator.interpolate

    def forward(self, x, T_out):
        """
        Args:
            x: (B, T_in, 1024) continuous semantic features
            T_out: scalar tensor, target output length
        Returns:
            out: (B, T_out, 512) length-regulated features
        """
        x = self.content_in_proj(x)
        if self.interpolate:
            x = F.interpolate(x.transpose(1, 2).contiguous(),
                              size=T_out, mode='nearest')
        else:
            x = x.transpose(1, 2).contiguous()
        torch._check(x.shape[2] > 0)
        out = self.model(x).transpose(1, 2).contiguous()
        return out


def export_onnx(output_path, device="cuda:0"):
    cfg = OmegaConf.load("checkpoints/config.yaml")
    s2mel_path = f"checkpoints/{cfg.s2mel_checkpoint}"
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(s2mel, None, s2mel_path,
                                       load_only_params=True, ignore_modules=[], is_distributed=False)
    s2mel = s2mel.to(device).eval()

    regulator = s2mel.models['length_regulator']
    wrapper = LengthRegulatorWrapper(regulator).eval().to(device)

    B, T_in, T_out_val = 2, 100, 172
    dummy_x = torch.randn(B, T_in, 1024, device=device)
    dummy_T_out = torch.tensor(T_out_val, device=device)

    with torch.no_grad():
        out = wrapper(dummy_x, dummy_T_out)
    print(f"Output shape: {out.shape}")

    B_dim = torch.export.Dim("B", min=1, max=8)
    T_in_dim = torch.export.Dim("T_in", min=10, max=2000)
    dynamic_shapes = {
        "x": {0: B_dim, 1: T_in_dim},
        "T_out": None,
    }

    print("Exporting length_regulator to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_x, dummy_T_out),
            output_path,
            input_names=["x", "T_out"],
            output_names=["out"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    print(f"ONNX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="length_regulator.onnx")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_onnx(args.output, args.device)


if __name__ == "__main__":
    main()
