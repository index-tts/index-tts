# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ALL_COMPLETED
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from einops import rearrange, repeat

from indextts.utils.maskgct.models.codec.amphion_codec.quantize import ResidualVQ
from indextts.utils.maskgct.models.codec.kmeans.vocos import VocosBackbone


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def compute_codebook_perplexity(indices, codebook_size):
    indices = indices.flatten()
    prob = torch.bincount(indices, minlength=codebook_size).float() / indices.size(0)
    perp = torch.exp(-torch.sum(prob * torch.log(prob + 1e-10)))
    return perp


class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        downsample_scale=1,
        cfg=None,
    ):
        super().__init__()
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
        hidden_size = (
            cfg.hidden_size
            if cfg is not None and hasattr(cfg, "hidden_size")
            else hidden_size
        )
        vocos_dim = (
            cfg.vocos_dim
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_dim
        )
        vocos_intermediate_dim = (
            cfg.vocos_intermediate_dim
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_intermediate_dim
        )
        vocos_num_layers = (
            cfg.vocos_num_layers
            if cfg is not None and hasattr(cfg, "vocos_dim")
            else vocos_num_layers
        )
        num_quantizers = (
            cfg.num_quantizers
            if cfg is not None and hasattr(cfg, "num_quantizers")
            else num_quantizers
        )
        downsample_scale = (
            cfg.downsample_scale
            if cfg is not None and hasattr(cfg, "downsample_scale")
            else downsample_scale
        )

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale

        if self.downsample_scale != None and self.downsample_scale > 1:
            self.down = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1
            )
            self.up = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1
            )

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )

        self.reset_parameters()

    def forward(self, x):

        # downsample
        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        # encoder
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        # vq
        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        # decoder
        x = self.decoder(quantized_out)

        # up
        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x_rec = self.up(x).transpose(1, 2)

        codebook_loss = (all_codebook_losses + all_commit_losses).mean()
        all_indices = all_indices

        return x_rec, codebook_loss, all_indices

    def quantize(self, x):

        if self.downsample_scale != None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self):
        self.apply(init_weights)


if __name__ == "__main__":
    repcodec = RepCodec(vocos_dim=1024, downsample_scale=2)
    print(repcodec)
    print(sum(p.numel() for p in repcodec.parameters()) / 1e6)
    x = torch.randn(5, 10, 1024)
    x_rec, codebook_loss, all_indices = repcodec(x)
    print(x_rec.shape, codebook_loss, all_indices.shape)
    vq_id, emb = repcodec.quantize(x)
    print(vq_id.shape, emb.shape)
