# Copyright (c) 2023 Amphion.
#
# This code is modified from https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/speechtokenizer/model.py
# Licensed under Apache License 2.0

from .modules.seanet import SEANetEncoder, SEANetDecoder
from .modules.quantization import ResidualVectorQuantizer
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np


class SpeechTokenizer(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        config : json
            Model Config.

        """
        super().__init__()
        self.encoder = SEANetEncoder(
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=config.get("strides"),
            lstm=config.get("lstm_layers"),
            bidirectional=config.get("bidirectional"),
            dilation_base=config.get("dilation_base"),
            residual_kernel_size=config.get("residual_kernel_size"),
            n_residual_layers=config.get("n_residual_layers"),
            activation=config.get("activation"),
        )
        self.sample_rate = config.get("sample_rate")
        self.n_q = config.get("n_q")
        self.downsample_rate = np.prod(config.get("strides"))
        if config.get("dimension") != config.get("semantic_dimension"):
            self.transform = nn.Linear(
                config.get("dimension"), config.get("semantic_dimension")
            )
        else:
            self.transform = nn.Identity()
        self.quantizer = ResidualVectorQuantizer(
            dimension=config.get("dimension"),
            n_q=config.get("n_q"),
            bins=config.get("codebook_size"),
        )
        self.decoder = SEANetDecoder(
            n_filters=config.get("n_filters"),
            dimension=config.get("dimension"),
            ratios=config.get("strides"),
            lstm=config.get("lstm_layers"),
            bidirectional=False,
            dilation_base=config.get("dilation_base"),
            residual_kernel_size=config.get("residual_kernel_size"),
            n_residual_layers=config.get("n_residual_layers"),
            activation=config.get("activation"),
        )

    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        """

        Parameters
        ----------
        config_path : str
            Path of model configuration file.
        ckpt_path : str
            Path of model  checkpoint.

        Returns
        -------
        model : SpeechTokenizer
            SpeechTokenizer model.

        """
        import json

        with open(config_path) as f:
            cfg = json.load(f)
        model = cls(cfg)
        params = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(params)
        return model

    def forward(self, x: torch.tensor, n_q: int = None, layers: list = [0]):
        """

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is the first layer.

        Returns
        -------
        o : torch.tensor
            Output wavs. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from residual vector quantizers.
        feature : torch.tensor
            Output of RVQ's first layer. Shape: (batch, timesteps, dimension)

        """
        n_q = n_q if n_q else self.n_q
        e = self.encoder(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(
            e, n_q=n_q, layers=layers
        )
        feature = rearrange(quantized_list[0], "b d t -> b t d")
        feature = self.transform(feature)
        o = self.decoder(quantized)
        return o, commit_loss, feature

    def forward_feature(self, x: torch.tensor, layers: list = None):
        """

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape should be (batch, channels, timesteps).
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is all layers.

        Returns
        -------
        quantized_list : list[torch.tensor]
            Quantized of required layers.

        """
        e = self.encoder(x)
        layers = layers if layers else list(range(self.n_q))
        quantized, codes, commit_loss, quantized_list = self.quantizer(e, layers=layers)
        return quantized_list

    def encode(self, x: torch.tensor, n_q: int = None, st: int = None):
        """

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (n_q, batch, timesteps)

        """
        e = self.encoder(x)
        if st is None:
            st = 0
        n_q = n_q if n_q else self.n_q
        codes = self.quantizer.encode(e, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.tensor, st: int = 0):
        """

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (n_q, batch, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        """
        quantized = self.quantizer.decode(codes, st=st)
        o = self.decoder(quantized)
        return o
