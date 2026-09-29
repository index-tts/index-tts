"""Export BigVGAN vocoder to ONNX.

Usage:
    python export_bigvgan_onnx.py [--output bigvgan.onnx] [--verify]
"""

import argparse
import torch
import numpy as np
from omegaconf import OmegaConf

from indextts.s2mel.modules.bigvgan import bigvgan


def load_bigvgan(cfg_path="checkpoints/config.yaml", device="cuda:0"):
    cfg = OmegaConf.load(cfg_path)
    model = bigvgan.BigVGAN.from_pretrained(
        cfg.vocoder.name, use_cuda_kernel=False
    )
    model = model.to(device)
    model.remove_weight_norm()
    model.eval()
    return model


def export_onnx(model, output_path, device="cuda:0"):
    model.eval()

    T_mel = 1024
    B = 2
    x = torch.randn(B, 80, T_mel, device=device)

    B_dim = torch.export.Dim("B", min=1, max=32)
    T_dim = torch.export.Dim("T_mel", min=64, max=8192)
    dynamic_shapes = {"x": {0: B_dim, 2: T_dim}}

    print("Exporting BigVGAN to ONNX via dynamo...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,),
            output_path,
            input_names=["mel"],
            output_names=["audio"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    print(f"ONNX model saved to {output_path}")


def verify_onnx(model, onnx_path, device="cuda:0"):
    import onnxruntime as ort

    model.eval()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    for T_mel in [256, 512, 1024]:
        x = torch.randn(1, 80, T_mel, device=device)

        with torch.no_grad():
            pt_out = model(x)

        ort_inputs = {"mel": x.cpu().numpy()}
        ort_out = sess.run(None, ort_inputs)[0]

        pt_np = pt_out.cpu().numpy()
        max_diff = np.max(np.abs(pt_np - ort_out))
        mean_diff = np.mean(np.abs(pt_np - ort_out))
        print(f"  T_mel={T_mel}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
              f"out_shape={ort_out.shape}")

        if max_diff > 0.01:
            print(f"  WARNING: large numerical difference!")
        else:
            print(f"  OK")


def main():
    parser = argparse.ArgumentParser(description="Export BigVGAN to ONNX")
    parser.add_argument("--cfg", default="checkpoints/config.yaml")
    parser.add_argument("--output", default="bigvgan.onnx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    model = load_bigvgan(args.cfg, args.device)
    export_onnx(model, args.output, args.device)

    if args.verify:
        model_fresh = load_bigvgan(args.cfg, args.device)
        verify_onnx(model_fresh, args.output, args.device)


if __name__ == "__main__":
    main()
