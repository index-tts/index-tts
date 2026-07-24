"""Export the W2V-BERT speech semantic encoder to ONNX.

Usage:
    python export_speech_semantic_encoder_onnx.py [--output speech_semantic_encoder.onnx]
"""

import argparse
import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel


class SpeechSemanticEncoder(nn.Module):
    """Wrapper that runs W2V-BERT up to layer 17 and returns that hidden state."""

    def __init__(self, full_model):
        super().__init__()
        self.feature_projection = full_model.feature_projection
        self.layers = nn.ModuleList(full_model.encoder.layers[:17])
        self.config = full_model.config

    def forward(self, input_features, attention_mask):
        hidden_states = self.feature_projection(input_features)[0]

        B, T, _ = hidden_states.shape
        expand_mask = attention_mask[:, None, None, :].expand(B, 1, T, T).to(hidden_states.dtype)
        expand_mask = (1.0 - expand_mask) * torch.finfo(hidden_states.dtype).min

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=expand_mask)[0]

        return hidden_states


def export_onnx(output_path, device="cuda:0"):
    print("Loading W2V-BERT model...")
    full_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    full_model.eval().to(device)

    wrapper = SpeechSemanticEncoder(full_model).eval().to(device)
    del full_model
    torch.cuda.empty_cache()

    print("Verifying wrapper output matches original...")

    B, T_feat = 2, 200
    dummy_features = torch.randn(B, T_feat, 160, device=device)
    dummy_mask = torch.ones(B, T_feat, dtype=torch.long, device=device)

    with torch.no_grad():
        wrapper_out = wrapper(dummy_features, dummy_mask.float())
    print(f"  Wrapper output shape: {wrapper_out.shape}")

    B_dim = torch.export.Dim("B", min=1, max=8)
    T_dim = torch.export.Dim("T_feat", min=50, max=3000)
    dynamic_shapes = {
        "input_features": {0: B_dim, 1: T_dim},
        "attention_mask": {0: B_dim, 1: T_dim},
    }

    print(f"Exporting to ONNX via dynamo...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_mask.float()),
            output_path,
            input_names=["input_features", "attention_mask"],
            output_names=["hidden_states_17"],
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )
    print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="speech_semantic_encoder.onnx")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    export_onnx(args.output, args.device)


if __name__ == "__main__":
    main()
