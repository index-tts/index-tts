# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This code is modified from https://github.com/sh-lee-prml/HierSpeechpp/blob/main/ttv_v1/styleencoder.py

from . import attentions
from torch import nn
import torch
from torch.nn import functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size=kernel_size, padding=2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class StyleEncoder(torch.nn.Module):
    def __init__(self, in_dim=513, hidden_dim=128, out_dim=256):

        super().__init__()

        self.in_dim = in_dim  # Linear 513 wav2vec 2.0 1024
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = attentions.MultiHeadAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.n_head,
            p_dropout=self.dropout,
            proximal_bias=False,
            proximal_init=True,
        )
        self.atten_drop = nn.Dropout(self.dropout)
        self.fc = nn.Conv1d(self.hidden_dim, self.out_dim, 1)

    def forward(self, x, mask=None):

        # spectral
        x = self.spectral(x) * mask
        # temporal
        x = self.temporal(x) * mask

        # self-attention
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        y = self.slf_attn(x, x, attn_mask=attn_mask)
        x = x + self.atten_drop(y)

        # fc
        x = self.fc(x)

        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=2)
        else:
            len_ = mask.sum(dim=2)
            x = x.sum(dim=2)

            out = torch.div(x, len_)
        return out
