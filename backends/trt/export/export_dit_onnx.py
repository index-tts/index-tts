"""Export the DiT estimator to ONNX.

Usage:
    python export_dit_onnx.py [--output dit.onnx] [--opset 17] [--verify]
"""

import argparse
import torch
import numpy as np
from omegaconf import OmegaConf

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2


def load_dit_model(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                   device="cuda:0"):
    cfg = OmegaConf.load(cfg_path)
    s2mel_path = f"{model_dir}/{cfg.s2mel_checkpoint}"

    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel, None, s2mel_path,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    s2mel = s2mel.to(device).eval()

    estimator = s2mel.models["cfm"].estimator
    estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    return estimator


def remove_weight_norm_from_model(model):
    """Remove weight_norm parametrizations so the model exports cleanly."""
    for module in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(module)
        except ValueError:
            pass


def _plain_fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    """Plain Python replacement for the @torch.jit.script version
    that causes issues with the ONNX JIT tracer."""
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


def patch_jit_script_functions():
    """Replace @torch.jit.script functions with plain Python equivalents."""
    import indextts.s2mel.modules.commons as commons_mod
    from indextts.s2mel.modules import wavenet as wavenet_mod
    original_commons = commons_mod.fused_add_tanh_sigmoid_multiply
    original_wavenet = wavenet_mod.commons.fused_add_tanh_sigmoid_multiply
    commons_mod.fused_add_tanh_sigmoid_multiply = _plain_fused_add_tanh_sigmoid_multiply
    wavenet_mod.commons.fused_add_tanh_sigmoid_multiply = _plain_fused_add_tanh_sigmoid_multiply
    return (original_commons, original_wavenet)


def restore_jit_script_functions(originals):
    import indextts.s2mel.modules.commons as commons_mod
    from indextts.s2mel.modules import wavenet as wavenet_mod
    commons_mod.fused_add_tanh_sigmoid_multiply = originals[0]
    wavenet_mod.commons.fused_add_tanh_sigmoid_multiply = originals[1]


class DiTWrapper(torch.nn.Module):
    """Wrapper that converts bool operations to ONNX-friendly ones."""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, prompt_x, x_lens, t, style, mu):
        return self.estimator(x, prompt_x, x_lens, t, style, mu)


def export_onnx(estimator, output_path, opset_version=17, device="cuda:0"):
    remove_weight_norm_from_model(estimator)
    estimator.eval()

    originals = patch_jit_script_functions()

    try:
        T = 1024
        B = 2
        x = torch.randn(B, 80, T, device=device)
        prompt_x = torch.randn(B, 80, T, device=device)
        x_lens = torch.tensor([T, T], dtype=torch.long, device=device)  # (B,)
        t = torch.tensor([0.5, 0.5], device=device)
        style = torch.randn(B, 192, device=device)
        mu = torch.randn(B, T, 512, device=device)

        B_dim = torch.export.Dim("B", min=1, max=32)
        T_dim = torch.export.Dim("T", min=64, max=8192)
        dynamic_shapes = {
            "x":        {0: B_dim, 2: T_dim},
            "prompt_x": {0: B_dim, 2: T_dim},
            "x_lens":   {0: B_dim},
            "t":        {0: B_dim},
            "style":    {0: B_dim},
            "cond":     {0: B_dim, 1: T_dim},
        }

        print(f"Exporting DiT to ONNX via dynamo (opset {opset_version})...")
        with torch.no_grad():
            export_output = torch.onnx.export(
                estimator,
                (x, prompt_x, x_lens, t, style, mu),
                output_path,
                input_names=["x", "prompt_x", "x_lens", "t", "style", "cond"],
                output_names=["output"],
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
            )
        print(f"ONNX model saved to {output_path}")
    finally:
        restore_jit_script_functions(originals)


def verify_onnx(estimator, onnx_path, device="cuda:0"):
    import onnxruntime as ort

    remove_weight_norm_from_model(estimator)
    estimator.eval()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    for T in [256, 1024, 2048]:
        B = 2
        x = torch.randn(B, 80, T, device=device)
        prompt_x = torch.randn(B, 80, T, device=device)
        x_lens = torch.full((B,), T, dtype=torch.long, device=device)
        t = torch.tensor([0.5, 0.5], device=device)
        style = torch.randn(B, 192, device=device)
        mu = torch.randn(B, T, 512, device=device)

        with torch.no_grad():
            pt_out = estimator(x, prompt_x, x_lens, t, style, mu)

        ort_inputs = {
            "x": x.cpu().numpy(),
            "prompt_x": prompt_x.cpu().numpy(),
            "x_lens": x_lens.cpu().numpy(),
            "t": t.cpu().numpy(),
            "style": style.cpu().numpy(),
            "cond": mu.cpu().numpy(),
        }
        ort_out = sess.run(None, ort_inputs)[0]

        pt_np = pt_out.cpu().numpy()
        max_diff = np.max(np.abs(pt_np - ort_out))
        mean_diff = np.mean(np.abs(pt_np - ort_out))
        print(f"  T={T}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
              f"output_shape={ort_out.shape}")

        if max_diff > 0.01:
            print(f"  WARNING: large numerical difference at T={T}!")
        else:
            print(f"  OK")


def main():
    parser = argparse.ArgumentParser(description="Export DiT estimator to ONNX")
    parser.add_argument("--cfg", default="checkpoints/config.yaml")
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--output", default="dit.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output against PyTorch")
    args = parser.parse_args()

    estimator = load_dit_model(args.cfg, args.model_dir, args.device)

    export_onnx(estimator, args.output, args.opset, args.device)

    if args.verify:
        estimator_fresh = load_dit_model(args.cfg, args.model_dir, args.device)
        verify_onnx(estimator_fresh, args.output, args.device)


if __name__ == "__main__":
    main()
