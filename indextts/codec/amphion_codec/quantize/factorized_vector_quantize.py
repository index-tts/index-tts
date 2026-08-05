# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.005,
        codebook_loss_weight=1.0,
        use_l2_normlize=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z):
        """
        Parameters
        ----------
        z: torch.Tensor[B x D x T]

        Returns
        -------
        z_q: torch.Tensor[B x D x T]
            Quantized continuous representation of input
        commit_loss: Tensor[B]
            Commitment loss to train encoder to predict vectors closer to codebook entries
        codebook_loss: Tensor[B]
            Codebook loss to update the codebook
        indices: torch.Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        z_e: torch.Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes project input into low-dimensional space if self.input_dim != self.codebook_dim
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)

        # Compute commitment loss and codebook loss
        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_project(z_q)

        return z_q, commit_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # if use_l2_normlize is True, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices

    def vq2emb(self, vq, out_proj=True):
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def latent2dist(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # if use_l2_normlize is True, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )  # (b*t, k)

        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        dist = rearrange(dist, "(b t) k -> b t k", b=latents.size(0))
        z_q = self.decode_code(indices)

        return -dist, indices, z_q
