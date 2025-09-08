from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional

import torch
import yaml
from torch import nn
from .heads import ISTFTHead
from .models import VocosBackbone


class Vocos(nn.Module):
    """
    The Vocos class represents a Fourier-based neural vocoder for audio synthesis.
    This class is primarily designed for inference, with support for loading from pretrained
    model checkpoints. It consists of three main components: a feature extractor,
    a backbone, and a head.
    """

    def __init__(
        self, args,
    ):
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels=args.vocos.backbone.input_channels,
            dim=args.vocos.backbone.dim,
            intermediate_dim=args.vocos.backbone.intermediate_dim,
            num_layers=args.vocos.backbone.num_layers,
        )
        self.head = ISTFTHead(
            dim=args.vocos.head.dim,
            n_fft=args.vocos.head.n_fft,
            hop_length=args.vocos.head.hop_length,
            padding=args.vocos.head.padding,
        )

    def forward(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                                     C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output
