# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from .fvq import FactorizedVectorQuantize


class ResidualVQ(nn.Module):
    """Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(self, *, num_quantizers, codebook_size, **kwargs):
        super().__init__()
        VQ = FactorizedVectorQuantize
        if type(codebook_size) == int:
            codebook_size = [codebook_size] * num_quantizers
        self.layers = nn.ModuleList(
            [VQ(codebook_size=2**size, **kwargs) for size in codebook_size]
        )
        self.num_quantizers = num_quantizers
        self.quantizer_dropout = kwargs.get("quantizer_dropout", 0.0)
        self.dropout_type = kwargs.get("dropout_type", None)

    def forward(self, x, n_quantizers=None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        if self.training:
            n_quantizers = torch.ones((x.shape[0],)) * self.num_quantizers + 1
            if self.dropout_type == "linear":
                dropout = torch.randint(1, self.num_quantizers + 1, (x.shape[0],))
            elif self.dropout_type == "exp":
                dropout = torch.randint(
                    1, int(math.log2(self.num_quantizers)), (x.shape[0],)
                )
                dropout = torch.pow(2, dropout)
            n_dropout = int(x.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(x.device)

        for idx, layer in enumerate(self.layers):
            if not self.training and idx >= n_quantizers:
                break
            quantized, indices, loss = layer(residual)

            mask = (
                torch.full((x.shape[0],), fill_value=idx, device=x.device)
                < n_quantizers
            )

            residual = residual - quantized

            quantized_out = quantized_out + quantized * mask[:, None, None]

            # loss
            loss = (loss * mask).mean()

            all_indices.append(indices)
            all_losses.append(loss)
            all_quantized.append(quantized)
        all_losses, all_indices, all_quantized = map(
            torch.stack, (all_losses, all_indices, all_quantized)
        )
        return quantized_out, all_indices, all_losses, all_quantized

    def vq2emb(self, vq):
        # vq: [n_quantizers, B, T]
        quantized_out = 0.0
        for idx, layer in enumerate(self.layers):
            quantized = layer.vq2emb(vq[idx])
            quantized_out += quantized
        return quantized_out

    def get_emb(self):
        embs = []
        for idx, layer in enumerate(self.layers):
            embs.append(layer.get_emb())
        return embs
