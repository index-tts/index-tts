# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from indextts.utils.maskgct.models.codec.amphion_codec.quantize.factorized_vector_quantize import (
    FactorizedVectorQuantize,
)
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.vector_quantize import VectorQuantize
from indextts.utils.maskgct.models.codec.amphion_codec.quantize.lookup_free_quantize import LookupFreeQuantize


class ResidualVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "vq",  # "vq" or "fvq" or "lfq"
        quantizer_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type
        self.quantizer_dropout = quantizer_dropout

        if quantizer_type == "vq":
            VQ = VectorQuantize
        elif quantizer_type == "fvq":
            VQ = FactorizedVectorQuantize
        elif quantizer_type == "lfq":
            VQ = LookupFreeQuantize
        else:
            raise ValueError(f"Unknown quantizer type {quantizer_type}")

        self.quantizers = nn.ModuleList(
            [
                VQ(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z, n_quantizers: int = None):
        """
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        "quantized_out" : Tensor[B x D x T]
            Quantized continuous representation of input
        "all_indices" : Tensor[N x B x T]
            Codebook indices for each codebook
            (quantized discrete representation of input)
        "all_commit_losses" : Tensor[N]
        "all_codebook_losses" : Tensor[N]
        "all_quantized" : Tensor[N x B x D x T]
        """

        quantized_out = 0.0
        residual = z

        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commit_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            quantized_out = quantized_out + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commit_loss_i = (commit_loss_i * mask).mean()
            codebook_loss_i = (codebook_loss_i * mask).mean()

            all_commit_losses.append(commit_loss_i)
            all_codebook_losses.append(codebook_loss_i)
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        all_commit_losses, all_codebook_losses, all_indices, all_quantized = map(
            torch.stack,
            (all_commit_losses, all_codebook_losses, all_indices, all_quantized),
        )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )

    def vq2emb(self, vq, n_quantizers=None):
        quantized_out = 0.0
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        for idx, quantizer in enumerate(self.quantizers):
            if idx >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[idx])
        return quantized_out

    def latent2dist(self, z, n_quantizers=None):
        quantized_out = 0.0
        residual = z

        all_dists = []
        all_indices = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            dist_i, indices_i, z_q_i = quantizer.latent2dist(residual)
            all_dists.append(dist_i)
            all_indices.append(indices_i)

            quantized_out = quantized_out + z_q_i
            residual = residual - z_q_i

        all_dists = torch.stack(all_dists)
        all_indices = torch.stack(all_indices)

        return all_dists, all_indices
