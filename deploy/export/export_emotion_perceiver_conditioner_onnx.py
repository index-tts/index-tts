"""Export emotion perceiver conditioner to ONNX.

Usage:
    python export_emotion_perceiver_conditioner_onnx.py [--output emotion_perceiver_conditioner.onnx]
"""

import argparse
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.gpt.perceiver import Attend
from indextts.gpt.conformer.attention import MultiHeadedAttention
from indextts.utils.checkpoint import load_checkpoint

_MASK_VALUE = -1e4
_ORIGINAL_ATTEND_FORWARD = Attend.forward
_ORIGINAL_MHA_FORWARD_ATTENTION = MultiHeadedAttention.forward_attention


def _patch_attend_class_for_fp16():
    """Patch attention classes to use FP16-safe mask value (-1e4 instead of -inf)."""
    from einops import rearrange

    def _fp16_safe_attend_forward(self, q, k, v, mask=None):
        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"
        sim = torch.einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, _MASK_VALUE)

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, _MASK_VALUE)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)
        return out

    def _fp16_safe_mha_forward_attention(self, value, scores, mask=torch.ones((0, 0, 0), dtype=torch.bool)):
        n_batch = value.size(0)
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(mask, _MASK_VALUE)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    Attend.forward = _fp16_safe_attend_forward
    MultiHeadedAttention.forward_attention = _fp16_safe_mha_forward_attention


def _restore_attend_class():
    """Restore the original forwards."""
    Attend.forward = _ORIGINAL_ATTEND_FORWARD
    MultiHeadedAttention.forward_attention = _ORIGINAL_MHA_FORWARD_ATTENTION


class EmotionPerceiverConditioner(nn.Module):
    """Wraps emo ConformerEncoder + emo PerceiverResampler + Linear layers for ONNX export."""

    def __init__(self, gpt_model):
        super().__init__()
        self.emo_conditioning_encoder = gpt_model.emo_conditioning_encoder
        self.emo_perceiver_encoder = gpt_model.emo_perceiver_encoder
        self.emo_cond_mask_pad = gpt_model.emo_cond_mask_pad
        self.emovec_layer = gpt_model.emovec_layer
        self.emo_layer = gpt_model.emo_layer

    def forward(self, speech_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speech_input: (B, T, 1024) padded input features
            lengths: (B,) actual per-sample sequence lengths
        Returns:
            emo_vec: (B, 1280) emotion vector
        """
        encoder_out, mask = self.emo_conditioning_encoder(speech_input, lengths)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        emo_cond = self.emo_perceiver_encoder(encoder_out, conds_mask)
        emo_cond = emo_cond.squeeze(1)
        emo_vec = self.emovec_layer(emo_cond)
        emo_vec = self.emo_layer(emo_vec)
        return emo_vec


def export_onnx(output_path, device="cuda:0"):
    _patch_attend_class_for_fp16()

    cfg = OmegaConf.load("checkpoints/config.yaml")
    print("Loading GPT model...")
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, f"checkpoints/{cfg.gpt_checkpoint}")
    gpt = gpt.to(device).eval()

    wrapper = EmotionPerceiverConditioner(gpt).eval()
    del gpt
    torch.cuda.empty_cache()

    B, T = 2, 300
    dummy_input = torch.randn(B, T, 1024, device=device)
    dummy_lengths = torch.tensor([T, T - 50], device=device, dtype=torch.int64)

    with torch.no_grad():
        out = wrapper(dummy_input, dummy_lengths)
    print(f"  Output shape: {out.shape}")

    B_dim = torch.export.Dim("B", min=1, max=8)
    T_dim = torch.export.Dim("T", min=3, max=1500)
    dynamic_shapes = {
        "speech_input": {0: B_dim, 1: T_dim},
        "lengths": {0: B_dim},
    }

    print("Exporting to ONNX via dynamo...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input, dummy_lengths),
            output_path,
            input_names=["speech_input", "lengths"],
            output_names=["emo_vec"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="emotion_perceiver_conditioner.onnx")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_onnx(args.output, args.device)


if __name__ == "__main__":
    main()
