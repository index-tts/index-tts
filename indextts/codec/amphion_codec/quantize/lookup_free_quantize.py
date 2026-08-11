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


class LookupFreeQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        assert 2**codebook_dim == codebook_size

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

    def forward(self, z):
        z_e = self.in_project(z)
        z_e = F.sigmoid(z_e)

        z_q = z_e + (torch.round(z_e) - z_e).detach()

        z_q = self.out_project(z_q)

        commit_loss = torch.zeros(z.shape[0], device=z.device)
        codebook_loss = torch.zeros(z.shape[0], device=z.device)

        bits = (
            2
            ** torch.arange(self.codebook_dim, device=z.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .long()
        )  # (1, d, 1)
        indices = (torch.round(z_e.clone().detach()).long() * bits).sum(1).long()

        return z_q, commit_loss, codebook_loss, indices, z_e

    def vq2emb(self, vq, out_proj=True):
        emb = torch.zeros(
            vq.shape[0], self.codebook_dim, vq.shape[-1], device=vq.device
        )  # (B, d, T)
        for i in range(self.codebook_dim):
            emb[:, i, :] = (vq % 2).float()
            vq = vq // 2
        if out_proj:
            emb = self.out_project(emb)
        return emb
