# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)


import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """Vector quantization w/ exponential moving averages (EMA)"""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay=0.8,
        commitment=1.0,
        eps=1e-5,
        n_embed=None,
    ):
        super().__init__()
        n_embed = self.default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def exists(self, val):
        return val is not None

    def default(self, val, d):
        return val if self.exists(val) else d

    def ema_inplace(self, moving_avg, new, decay):
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def laplace_smoothing(self, x, n_categories, eps=1e-5):
        return (x + eps) / (x.sum() + n_categories * eps)

    def forward(self, input):
        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            self.ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = (
                self.laplace_smoothing(self.cluster_size, self.n_embed, self.eps)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()

        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize, loss, perplexity

    def forward_index(self, input):
        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))
        quantize = input + (quantize - input).detach()

        return quantize, embed_ind


class ResidualVQ(nn.Module):
    """Residual VQ following algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantize(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x):
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_perplexities = []
        for layer in self.layers:
            quantized, loss, perplexity = layer(residual)
            # Issue: https://github.com/lucidrains/vector-quantize-pytorch/issues/33
            # We found considering only the 1st layer VQ's graident results in better performance
            # residual = residual - quantized.detach() # considering all layers' graidents
            residual = (
                residual - quantized
            )  # considering only the first layer's graident
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_perplexities.append(perplexity)
        all_losses, all_perplexities = map(torch.stack, (all_losses, all_perplexities))
        return quantized_out, all_losses, all_perplexities

    def forward_index(self, x, flatten_idx=False):
        """
        all_indices: [num_of_quantizers, B, T]
        """
        quantized_out = 0.0
        residual = x
        all_indices = []
        for i, layer in enumerate(self.layers):
            quantized, indices = layer.forward_index(residual)
            # residual = residual - quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            if flatten_idx:
                indices += self.codebook_size * i
            all_indices.append(indices)
        all_indices = torch.stack(all_indices)
        return quantized_out, all_indices

    def initial(self):
        self.codebook = []
        for layer in self.layers:
            self.codebook.append(layer.codebook)
        self.codebook_size = self.codebook[0].size(0)
        self.codebook = torch.stack(self.codebook)
        self.codebook = self.codebook.reshape(-1, self.codebook.size(-1))

    def lookup(self, indices):
        quantized_out = F.embedding(indices, self.codebook)  # Num x T x C
        return torch.sum(quantized_out, dim=0, keepdim=True)


class Quantizer(nn.Module):
    def __init__(
        self,
        code_dim: int,
        codebook_num: int,
        codebook_size: int,
    ):
        super().__init__()
        self.codebook = ResidualVQ(
            dim=code_dim, num_quantizers=codebook_num, codebook_size=codebook_size
        )

    def initial(self):
        self.codebook.initial()

    def forward(self, z):
        zq, vqloss, perplexity = self.codebook(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, vqloss, perplexity

    def inference(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, indices

    def encode(self, z):
        zq, indices = self.codebook.forward_index(z.transpose(2, 1), flatten_idx=True)
        return zq, indices

    def decode(self, indices):
        z = self.codebook.lookup(indices)
        return z


class Conv1d1x1(nn.Conv1d):
    """1x1 Conv1d."""

    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, bias=bias
        )


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = -1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x


class ConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding=-1,
        output_padding=-1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride + 1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C', T').
        """
        x = self.deconv(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        dilation=1,
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={},
    ):
        super().__init__()
        self.activation = getattr(nn, nonlinear_activation)(
            **nonlinear_activation_params
        )
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias)

    def forward(self, x):
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class Projector(nn.Module):
    def __init__(
        self, input_channels: int, code_dim: int, kernel_size=3, stride=1, bias=False
    ):
        super().__init__()
        self.project = Conv1d(
            input_channels, code_dim, kernel_size=kernel_size, stride=stride, bias=bias
        )

    def forward(self, x):
        return self.project(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations=(1, 1),
        unit_kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.res_units = torch.nn.ModuleList()
        for dilation in dilations:
            self.res_units += [
                ResidualUnit(
                    in_channels,
                    in_channels,
                    kernel_size=unit_kernel_size,
                    dilation=dilation,
                )
            ]
        self.num_res = len(self.res_units)

        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                3 if stride == 1 else (2 * stride)
            ),  # special case: stride=1, do not use kernel=2
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv = Conv1d(
            in_channels=input_channels,
            out_channels=encode_channels,
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )
        self.conv_blocks = torch.nn.ModuleList()
        in_channels = encode_channels
        for idx, stride in enumerate(strides):
            out_channels = int(encode_channels * channel_ratios[idx])  # could be float
            self.conv_blocks += [
                EncoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            ]
            in_channels = out_channels
        self.num_blocks = len(self.conv_blocks)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block (no up-sampling)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dilations=(1, 1),
        unit_kernel_size=3,
        bias=True,
    ):
        super().__init__()

        if stride == 1:
            self.conv = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,  # fix kernel=3 when stride=1 for unchanged shape
                stride=stride,
                bias=bias,
            )
        else:
            self.conv = ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(2 * stride),
                stride=stride,
                bias=bias,
            )

        self.res_units = torch.nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            self.res_units += [
                ResidualUnit(
                    out_channels,
                    out_channels,
                    kernel_size=unit_kernel_size,
                    dilation=dilation,
                )
            ]
        self.num_res = len(self.res_units)

    def forward(self, x):
        x = self.conv(x)
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        output_channels: int,
        decode_channels: int,
        channel_ratios=(1, 1),
        strides=(1, 1),
        kernel_size=3,
        bias=True,
        block_dilations=(1, 1),
        unit_kernel_size=3,
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)

        self.conv1 = Conv1d(
            in_channels=code_dim,
            out_channels=int(decode_channels * channel_ratios[0]),
            kernel_size=kernel_size,
            stride=1,
            bias=False,
        )

        self.conv_blocks = torch.nn.ModuleList()
        for idx, stride in enumerate(strides):
            in_channels = int(decode_channels * channel_ratios[idx])
            if idx < (len(channel_ratios) - 1):
                out_channels = int(decode_channels * channel_ratios[idx + 1])
            else:
                out_channels = decode_channels
            self.conv_blocks += [
                DecoderBlock(
                    in_channels,
                    out_channels,
                    stride,
                    dilations=block_dilations,
                    unit_kernel_size=unit_kernel_size,
                    bias=bias,
                )
            ]
        self.num_blocks = len(self.conv_blocks)

        self.conv2 = Conv1d(out_channels, output_channels, kernel_size, 1, bias=False)

    def forward(self, z):
        x = self.conv1(z)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        x = self.conv2(x)
        return x


class VevoRepCodec(nn.Module):
    def __init__(
        self,
        input_channels=768,
        output_channels=768,
        encode_channels=768,
        decode_channels=768,
        code_dim=768,
        codebook_num=1,
        codebook_size=1024,
        bias=True,
        enc_ratios=(1, 1),
        dec_ratios=(1, 1),
        enc_strides=(1, 1),
        dec_strides=(1, 1),
        enc_kernel_size=3,
        dec_kernel_size=3,
        enc_block_dilations=(1, 1),
        enc_block_kernel_size=3,
        dec_block_dilations=(1, 1),
        dec_block_kernel_size=3,
    ):
        super().__init__()

        self.input_channels = input_channels

        self.encoder = Encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            channel_ratios=enc_ratios,
            strides=enc_strides,
            kernel_size=enc_kernel_size,
            bias=bias,
            block_dilations=enc_block_dilations,
            unit_kernel_size=enc_block_kernel_size,
        )

        self.decoder = Decoder(
            code_dim=code_dim,
            output_channels=output_channels,
            decode_channels=decode_channels,
            channel_ratios=dec_ratios,
            strides=dec_strides,
            kernel_size=dec_kernel_size,
            bias=bias,
            block_dilations=dec_block_dilations,
            unit_kernel_size=dec_block_kernel_size,
        )

        self.projector = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False,
        )

        self.quantizer = Quantizer(
            code_dim=code_dim, codebook_num=codebook_num, codebook_size=codebook_size
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.projector(x)
        zq, vqloss, perplexity = self.quantizer(z)
        y = self.decoder(zq)
        return y, zq, z, vqloss, perplexity
