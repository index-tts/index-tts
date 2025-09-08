# Copyright (c) 2023 Amphion.
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


class FactorizedVectorQuantize(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim, commitment, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment

        if dim != self.codebook_dim:
            self.in_proj = weight_norm(nn.Linear(dim, self.codebook_dim))
            self.out_proj = weight_norm(nn.Linear(self.codebook_dim, dim))
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        self._codebook = nn.Embedding(codebook_size, self.codebook_dim)

    @property
    def codebook(self):
        return self._codebook

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """
        # transpose since we use linear

        z = rearrange(z, "b d t -> b t d")

        # Factorized codes project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x T x D)
        z_e = rearrange(z_e, "b t d -> b d t")
        z_q, indices = self.decode_latents(z_e)

        if self.training:
            commitment_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
            commit_loss = commitment_loss + codebook_loss
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = rearrange(z_q, "b d t -> b t d")
        z_q = self.out_proj(z_q)
        z_q = rearrange(z_q, "b t d -> b d t")

        return z_q, indices, commit_loss

    def vq2emb(self, vq, proj=True):
        emb = self.embed_code(vq)
        if proj:
            emb = self.out_proj(emb)
        return emb.transpose(1, 2)

    def get_emb(self):
        return self.codebook.weight

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)
        # L2 normalize encodings and codebook
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
