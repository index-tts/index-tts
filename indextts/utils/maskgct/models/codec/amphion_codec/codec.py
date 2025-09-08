# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from indextts.utils.maskgct.models.codec.amphion_codec.quantize import (
    ResidualVQ,
    VectorQuantize,
    FactorizedVectorQuantize,
    LookupFreeQuantize,
)

from indextts.utils.maskgct.models.codec.amphion_codec.vocos import Vocos


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class CodecEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        up_ratios: list = [4, 5, 5, 6],
        out_channels: int = 256,
        use_tanh: bool = False,
        cfg=None,
    ):
        super().__init__()

        d_model = cfg.d_model if cfg is not None else d_model
        up_ratios = cfg.up_ratios if cfg is not None else up_ratios
        out_channels = cfg.out_channels if cfg is not None else out_channels
        use_tanh = cfg.use_tanh if cfg is not None else use_tanh

        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in up_ratios:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        if use_tanh:
            self.block += [nn.Tanh()]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def forward(self, x):
        return self.block(x)

    def reset_parameters(self):
        self.apply(init_weights)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class CodecDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        upsample_initial_channel: int = 1536,
        up_ratios: list = [5, 5, 4, 2],
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "vq",
        quantizer_dropout: float = 0.5,
        commitment: float = 0.25,
        codebook_loss_weight: float = 1.0,
        use_l2_normlize: bool = False,
        codebook_type: str = "euclidean",
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        eps: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        weight_init: bool = False,
        use_vocos: bool = False,
        vocos_dim: int = 384,
        vocos_intermediate_dim: int = 1152,
        vocos_num_layers: int = 8,
        n_fft: int = 800,
        hop_size: int = 200,
        padding: str = "same",
        cfg=None,
    ):
        super().__init__()

        in_channels = (
            cfg.in_channels
            if cfg is not None and hasattr(cfg, "in_channels")
            else in_channels
        )
        upsample_initial_channel = (
            cfg.upsample_initial_channel
            if cfg is not None and hasattr(cfg, "upsample_initial_channel")
            else upsample_initial_channel
        )
        up_ratios = (
            cfg.up_ratios
            if cfg is not None and hasattr(cfg, "up_ratios")
            else up_ratios
        )
        num_quantizers = (
            cfg.num_quantizers
            if cfg is not None and hasattr(cfg, "num_quantizers")
            else num_quantizers
        )
        codebook_size = (
            cfg.codebook_size
            if cfg is not None and hasattr(cfg, "codebook_size")
            else codebook_size
        )
        codebook_dim = (
            cfg.codebook_dim
            if cfg is not None and hasattr(cfg, "codebook_dim")
            else codebook_dim
        )
        quantizer_type = (
            cfg.quantizer_type
            if cfg is not None and hasattr(cfg, "quantizer_type")
            else quantizer_type
        )
        quantizer_dropout = (
            cfg.quantizer_dropout
            if cfg is not None and hasattr(cfg, "quantizer_dropout")
            else quantizer_dropout
        )
        commitment = (
            cfg.commitment
            if cfg is not None and hasattr(cfg, "commitment")
            else commitment
        )
        codebook_loss_weight = (
            cfg.codebook_loss_weight
            if cfg is not None and hasattr(cfg, "codebook_loss_weight")
            else codebook_loss_weight
        )
        use_l2_normlize = (
            cfg.use_l2_normlize
            if cfg is not None and hasattr(cfg, "use_l2_normlize")
            else use_l2_normlize
        )
        codebook_type = (
            cfg.codebook_type
            if cfg is not None and hasattr(cfg, "codebook_type")
            else codebook_type
        )
        kmeans_init = (
            cfg.kmeans_init
            if cfg is not None and hasattr(cfg, "kmeans_init")
            else kmeans_init
        )
        kmeans_iters = (
            cfg.kmeans_iters
            if cfg is not None and hasattr(cfg, "kmeans_iters")
            else kmeans_iters
        )
        decay = cfg.decay if cfg is not None and hasattr(cfg, "decay") else decay
        eps = cfg.eps if cfg is not None and hasattr(cfg, "eps") else eps
        threshold_ema_dead_code = (
            cfg.threshold_ema_dead_code
            if cfg is not None and hasattr(cfg, "threshold_ema_dead_code")
            else threshold_ema_dead_code
        )
        weight_init = (
            cfg.weight_init
            if cfg is not None and hasattr(cfg, "weight_init")
            else weight_init
        )
        use_vocos = (
            cfg.use_vocos
            if cfg is not None and hasattr(cfg, "use_vocos")
            else use_vocos
        )
        vocos_dim = (
            cfg.vocos_dim
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_dim
        )
        vocos_intermediate_dim = (
            cfg.vocos_intermediate_dim
            if cfg is not None and hasattr(cfg, "vocos_intermediate_dim")
            else vocos_intermediate_dim
        )
        vocos_num_layers = (
            cfg.vocos_num_layers
            if cfg is not None and hasattr(cfg, "vocos_num_layers")
            else vocos_num_layers
        )
        n_fft = cfg.n_fft if cfg is not None and hasattr(cfg, "n_fft") else n_fft
        hop_size = (
            cfg.hop_size if cfg is not None and hasattr(cfg, "hop_size") else hop_size
        )
        padding = (
            cfg.padding if cfg is not None and hasattr(cfg, "padding") else padding
        )

        if quantizer_type == "vq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
                quantizer_dropout=quantizer_dropout,
                commitment=commitment,
                codebook_loss_weight=codebook_loss_weight,
                use_l2_normlize=use_l2_normlize,
                codebook_type=codebook_type,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                decay=decay,
                eps=eps,
                threshold_ema_dead_code=threshold_ema_dead_code,
                weight_init=weight_init,
            )
        elif quantizer_type == "fvq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
                quantizer_dropout=quantizer_dropout,
                commitment=commitment,
                codebook_loss_weight=codebook_loss_weight,
                use_l2_normlize=use_l2_normlize,
            )
        elif quantizer_type == "lfq":
            self.quantizer = ResidualVQ(
                input_dim=in_channels,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_type=quantizer_type,
            )
        else:
            raise ValueError(f"Unknown quantizer type {quantizer_type}")

        if not use_vocos:
            # Add first conv layer
            channels = upsample_initial_channel
            layers = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]

            # Add upsampling + MRF blocks
            for i, stride in enumerate(up_ratios):
                input_dim = channels // 2**i
                output_dim = channels // 2 ** (i + 1)
                layers += [DecoderBlock(input_dim, output_dim, stride)]

            # Add final conv layer
            layers += [
                Snake1d(output_dim),
                WNConv1d(output_dim, 1, kernel_size=7, padding=3),
                nn.Tanh(),
            ]

            self.model = nn.Sequential(*layers)

        if use_vocos:
            self.model = Vocos(
                input_channels=in_channels,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                adanorm_num_embeddings=None,
                n_fft=n_fft,
                hop_size=hop_size,
                padding=padding,
            )

        self.reset_parameters()

    def forward(self, x=None, vq=False, eval_vq=False, n_quantizers=None):
        """
        if vq is True, x = encoder output, then return quantized output;
        else, x = quantized output, then return decoder output
        """
        if vq is True:
            if eval_vq:
                self.quantizer.eval()
            (
                quantized_out,
                all_indices,
                all_commit_losses,
                all_codebook_losses,
                all_quantized,
            ) = self.quantizer(x, n_quantizers=n_quantizers)
            return (
                quantized_out,
                all_indices,
                all_commit_losses,
                all_codebook_losses,
                all_quantized,
            )

        return self.model(x)

    def quantize(self, x, n_quantizers=None):
        self.quantizer.eval()
        quantized_out, vq, _, _, _ = self.quantizer(x, n_quantizers=n_quantizers)
        return quantized_out, vq

    # TODO: check consistency of vq2emb and quantize
    def vq2emb(self, vq, n_quantizers=None):
        return self.quantizer.vq2emb(vq, n_quantizers=n_quantizers)

    def decode(self, x):
        return self.model(x)

    def latent2dist(self, x, n_quantizers=None):
        return self.quantizer.latent2dist(x, n_quantizers=n_quantizers)

    def reset_parameters(self):
        self.apply(init_weights)
