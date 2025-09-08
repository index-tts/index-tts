# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.in_dim = normalized_shape
        self.norm = nn.LayerNorm(self.in_dim, eps=eps, elementwise_affine=False)
        self.style = nn.Linear(self.in_dim, self.in_dim * 2)
        self.style.bias.data[: self.in_dim] = 1
        self.style.bias.data[self.in_dim :] = 0

    def forward(self, x, condition):
        # x: (B, T, d); condition: (B, T, d)

        style = self.style(torch.mean(condition, dim=1, keepdim=True))

        gamma, beta = style.chunk(2, -1)

        out = self.norm(x)

        out = gamma * out + beta
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()

        self.dropout = dropout
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return F.dropout(x, self.dropout, training=self.training)


class TransformerFFNLayer(nn.Module):
    def __init__(
        self, encoder_hidden, conv_filter_size, conv_kernel_size, encoder_dropout
    ):
        super().__init__()

        self.encoder_hidden = encoder_hidden
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout

        self.ffn_1 = nn.Conv1d(
            self.encoder_hidden,
            self.conv_filter_size,
            self.conv_kernel_size,
            padding=self.conv_kernel_size // 2,
        )
        self.ffn_1.weight.data.normal_(0.0, 0.02)
        self.ffn_2 = nn.Linear(self.conv_filter_size, self.encoder_hidden)
        self.ffn_2.weight.data.normal_(0.0, 0.02)

    def forward(self, x):
        # x: (B, T, d)
        x = self.ffn_1(x.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # (B, T, d) -> (B, d, T) -> (B, T, d)
        x = F.relu(x)
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        encoder_hidden,
        encoder_head,
        conv_filter_size,
        conv_kernel_size,
        encoder_dropout,
        use_cln,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.encoder_head = encoder_head
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.encoder_dropout = encoder_dropout
        self.use_cln = use_cln

        if not self.use_cln:
            self.ln_1 = nn.LayerNorm(self.encoder_hidden)
            self.ln_2 = nn.LayerNorm(self.encoder_hidden)
        else:
            self.ln_1 = StyleAdaptiveLayerNorm(self.encoder_hidden)
            self.ln_2 = StyleAdaptiveLayerNorm(self.encoder_hidden)

        self.self_attn = nn.MultiheadAttention(
            self.encoder_hidden, self.encoder_head, batch_first=True
        )

        self.ffn = TransformerFFNLayer(
            self.encoder_hidden,
            self.conv_filter_size,
            self.conv_kernel_size,
            self.encoder_dropout,
        )

    def forward(self, x, key_padding_mask, conditon=None):
        # x: (B, T, d); key_padding_mask: (B, T), mask is 0; condition: (B, T, d)

        # self attention
        residual = x
        if self.use_cln:
            x = self.ln_1(x, conditon)
        else:
            x = self.ln_1(x)

        if key_padding_mask != None:
            key_padding_mask_input = ~(key_padding_mask.bool())
        else:
            key_padding_mask_input = None
        x, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask_input
        )
        x = F.dropout(x, self.encoder_dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        if self.use_cln:
            x = self.ln_2(x, conditon)
        else:
            x = self.ln_2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        enc_emb_tokens=None,
        encoder_layer=4,
        encoder_hidden=256,
        encoder_head=4,
        conv_filter_size=1024,
        conv_kernel_size=5,
        encoder_dropout=0.1,
        use_cln=False,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_cln = use_cln if use_cln is not None else cfg.use_cln

        if enc_emb_tokens != None:
            self.use_enc_emb = True
            self.enc_emb_tokens = enc_emb_tokens
        else:
            self.use_enc_emb = False

        self.position_emb = PositionalEncoding(
            self.encoder_hidden, self.encoder_dropout
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    self.encoder_hidden,
                    self.encoder_head,
                    self.conv_filter_size,
                    self.conv_kernel_size,
                    self.encoder_dropout,
                    self.use_cln,
                )
                for i in range(self.encoder_layer)
            ]
        )

        if self.use_cln:
            self.last_ln = StyleAdaptiveLayerNorm(self.encoder_hidden)
        else:
            self.last_ln = nn.LayerNorm(self.encoder_hidden)

    def forward(self, x, key_padding_mask, condition=None):
        if len(x.shape) == 2 and self.use_enc_emb:
            x = self.enc_emb_tokens(x)
            x = self.position_emb(x)
        else:
            x = self.position_emb(x)  # (B, T, d)

        for layer in self.layers:
            x = layer(x, key_padding_mask, condition)

        if self.use_cln:
            x = self.last_ln(x, condition)
        else:
            x = self.last_ln(x)

        return x
