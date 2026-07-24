"""Export CAMPPlus speaker style model to ONNX.

Usage:
    python export_campplus_onnx.py [--output campplus.onnx]
"""

import argparse
import torch
from huggingface_hub import hf_hub_download

from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus


def export_onnx(output_path, device="cuda:0"):
    ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    B, T = 1, 500
    dummy = torch.randn(B, T, 80, device=device)

    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")

    print("Exporting CAMPPlus to ONNX (classic trace)...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            input_names=["x"],
            output_names=["style"],
            dynamic_axes={"x": {0: "B", 1: "T"}, "style": {0: "B"}},
            opset_version=17,
        )
    print(f"ONNX saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="campplus.onnx")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_onnx(args.output, args.device)


if __name__ == "__main__":
    main()
