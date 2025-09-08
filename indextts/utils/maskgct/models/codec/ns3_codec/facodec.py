# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn, sin, pow
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .alias_free_torch import *
from .quantize import *
from einops import rearrange
from einops.layers.torch import Rearrange
from .transformer import TransformerEncoder
from .gradient_reversal import GradientReversal
from .melspec import MelSpectrogram


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CNNLSTM(nn.Module):
    def __init__(self, indim, outdim, head, global_pred=False):
        super().__init__()
        self.global_pred = global_pred
        self.model = nn.Sequential(
            ResidualUnit(indim, dilation=1),
            ResidualUnit(indim, dilation=2),
            ResidualUnit(indim, dilation=3),
            Activation1d(activation=SnakeBeta(indim, alpha_logscale=True)),
            Rearrange("b c t -> b t c"),
        )
        self.heads = nn.ModuleList([nn.Linear(indim, outdim) for i in range(head)])

    def forward(self, x):
        # x: [B, C, T]
        x = self.model(x)
        if self.global_pred:
            x = torch.mean(x, dim=1, keepdim=False)
        outs = [head(x) for head in self.heads]
        return outs


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta := x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Activation1d(activation=SnakeBeta(dim // 2, alpha_logscale=True)),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)


class FACodecEncoder(nn.Module):
    def __init__(
        self,
        ngf=32,
        up_ratios=(2, 4, 5, 5),
        out_channels=1024,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in up_ratios:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Activation1d(activation=SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Activation1d(activation=SnakeBeta(input_dim, alpha_logscale=True)),
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


class FACodecDecoder(nn.Module):
    def __init__(
        self,
        in_channels=256,
        upsample_initial_channel=1536,
        ngf=32,
        up_ratios=(5, 5, 4, 2),
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=1024,
        vq_commit_weight=0.005,
        vq_weight_init=False,
        vq_full_commit_loss=False,
        codebook_dim=8,
        codebook_size_prosody=10,  # true codebook size is equal to 2^codebook_size
        codebook_size_content=10,
        codebook_size_residual=10,
        quantizer_dropout=0.0,
        dropout_type="linear",
        use_gr_content_f0=False,
        use_gr_prosody_phone=False,
        use_gr_residual_f0=False,
        use_gr_residual_phone=False,
        use_gr_x_timbre=False,
        use_random_mask_residual=True,
        prob_random_mask_residual=0.75,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        self.use_random_mask_residual = use_random_mask_residual
        self.prob_random_mask_residual = prob_random_mask_residual

        self.vq_num_q_p = vq_num_q_p
        self.vq_num_q_c = vq_num_q_c
        self.vq_num_q_r = vq_num_q_r

        self.codebook_size_prosody = codebook_size_prosody
        self.codebook_size_content = codebook_size_content
        self.codebook_size_residual = codebook_size_residual

        quantizer_class = ResidualVQ

        self.quantizer = nn.ModuleList()

        # prosody
        quantizer = quantizer_class(
            num_quantizers=vq_num_q_p,
            dim=vq_dim,
            codebook_size=codebook_size_prosody,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
            quantizer_dropout=quantizer_dropout,
            dropout_type=dropout_type,
        )
        self.quantizer.append(quantizer)

        # phone
        quantizer = quantizer_class(
            num_quantizers=vq_num_q_c,
            dim=vq_dim,
            codebook_size=codebook_size_content,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
            quantizer_dropout=quantizer_dropout,
            dropout_type=dropout_type,
        )
        self.quantizer.append(quantizer)

        # residual
        if self.vq_num_q_r > 0:
            quantizer = quantizer_class(
                num_quantizers=vq_num_q_r,
                dim=vq_dim,
                codebook_size=codebook_size_residual,
                codebook_dim=codebook_dim,
                threshold_ema_dead_code=2,
                commitment=vq_commit_weight,
                weight_init=vq_weight_init,
                full_commit_loss=vq_full_commit_loss,
                quantizer_dropout=quantizer_dropout,
                dropout_type=dropout_type,
            )
            self.quantizer.append(quantizer)

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
            Activation1d(activation=SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.timbre_encoder = TransformerEncoder(
            enc_emb_tokens=None,
            encoder_layer=4,
            encoder_hidden=256,
            encoder_head=4,
            conv_filter_size=1024,
            conv_kernel_size=5,
            encoder_dropout=0.1,
            use_cln=False,
        )

        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.f0_predictor = CNNLSTM(in_channels, 1, 2)
        self.phone_predictor = CNNLSTM(in_channels, 5003, 1)

        self.use_gr_content_f0 = use_gr_content_f0
        self.use_gr_prosody_phone = use_gr_prosody_phone
        self.use_gr_residual_f0 = use_gr_residual_f0
        self.use_gr_residual_phone = use_gr_residual_phone
        self.use_gr_x_timbre = use_gr_x_timbre

        if self.vq_num_q_r > 0 and self.use_gr_residual_f0:
            self.res_f0_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 1, 2)
            )

        if self.vq_num_q_r > 0 and self.use_gr_residual_phone > 0:
            self.res_phone_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 5003, 1)
            )

        if self.use_gr_content_f0:
            self.content_f0_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 1, 2)
            )

        if self.use_gr_prosody_phone:
            self.prosody_phone_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 5003, 1)
            )

        if self.use_gr_x_timbre:
            self.x_timbre_predictor = nn.Sequential(
                GradientReversal(alpha=1),
                CNNLSTM(in_channels, 245200, 1, global_pred=True),
            )

        self.reset_parameters()

    def quantize(self, x, n_quantizers=None):
        outs, qs, commit_loss, quantized_buf = 0, [], [], []

        # prosody
        f0_input = x  # (B, d, T)
        f0_quantizer = self.quantizer[0]
        out, q, commit, quantized = f0_quantizer(f0_input, n_quantizers=n_quantizers)
        outs += out
        qs.append(q)
        quantized_buf.append(quantized.sum(0))
        commit_loss.append(commit)

        # phone
        phone_input = x
        phone_quantizer = self.quantizer[1]
        out, q, commit, quantized = phone_quantizer(
            phone_input, n_quantizers=n_quantizers
        )
        outs += out
        qs.append(q)
        quantized_buf.append(quantized.sum(0))
        commit_loss.append(commit)

        # residual
        if self.vq_num_q_r > 0:
            residual_quantizer = self.quantizer[2]
            residual_input = x - (quantized_buf[0] + quantized_buf[1]).detach()
            out, q, commit, quantized = residual_quantizer(
                residual_input, n_quantizers=n_quantizers
            )
            outs += out
            qs.append(q)
            quantized_buf.append(quantized.sum(0))  # [L, B, C, T] -> [B, C, T]
            commit_loss.append(commit)

        qs = torch.cat(qs, dim=0)
        commit_loss = torch.cat(commit_loss, dim=0)
        return outs, qs, commit_loss, quantized_buf

    def forward(
        self,
        x,
        vq=True,
        get_vq=False,
        eval_vq=True,
        speaker_embedding=None,
        n_quantizers=None,
        quantized=None,
    ):
        if get_vq:
            return self.quantizer.get_emb()
        if vq is True:
            if eval_vq:
                self.quantizer.eval()
            x_timbre = x
            outs, qs, commit_loss, quantized_buf = self.quantize(
                x, n_quantizers=n_quantizers
            )

            x_timbre = x_timbre.transpose(1, 2)
            x_timbre = self.timbre_encoder(x_timbre, None, None)
            x_timbre = x_timbre.transpose(1, 2)
            spk_embs = torch.mean(x_timbre, dim=2)
            return outs, qs, commit_loss, quantized_buf, spk_embs

        out = {}

        layer_0 = quantized[0]
        f0, uv = self.f0_predictor(layer_0)
        f0 = rearrange(f0, "... 1 -> ...")
        uv = rearrange(uv, "... 1 -> ...")

        layer_1 = quantized[1]
        (phone,) = self.phone_predictor(layer_1)

        out = {"f0": f0, "uv": uv, "phone": phone}

        if self.use_gr_prosody_phone:
            (prosody_phone,) = self.prosody_phone_predictor(layer_0)
            out["prosody_phone"] = prosody_phone

        if self.use_gr_content_f0:
            content_f0, content_uv = self.content_f0_predictor(layer_1)
            content_f0 = rearrange(content_f0, "... 1 -> ...")
            content_uv = rearrange(content_uv, "... 1 -> ...")
            out["content_f0"] = content_f0
            out["content_uv"] = content_uv

        if self.vq_num_q_r > 0:
            layer_2 = quantized[2]

            if self.use_gr_residual_f0:
                res_f0, res_uv = self.res_f0_predictor(layer_2)
                res_f0 = rearrange(res_f0, "... 1 -> ...")
                res_uv = rearrange(res_uv, "... 1 -> ...")
                out["res_f0"] = res_f0
                out["res_uv"] = res_uv

            if self.use_gr_residual_phone:
                (res_phone,) = self.res_phone_predictor(layer_2)
                out["res_phone"] = res_phone

        style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if self.vq_num_q_r > 0:
            if self.use_random_mask_residual:
                bsz = quantized[2].shape[0]
                res_mask = np.random.choice(
                    [0, 1],
                    size=bsz,
                    p=[
                        self.prob_random_mask_residual,
                        1 - self.prob_random_mask_residual,
                    ],
                )
                res_mask = (
                    torch.from_numpy(res_mask).unsqueeze(1).unsqueeze(1)
                )  # (B, 1, 1)
                res_mask = res_mask.to(
                    device=quantized[2].device, dtype=quantized[2].dtype
                )
                x = (
                    quantized[0].detach()
                    + quantized[1].detach()
                    + quantized[2] * res_mask
                )
                # x = quantized_perturbe[0].detach() + quantized[1].detach() + quantized[2] * res_mask
            else:
                x = quantized[0].detach() + quantized[1].detach() + quantized[2]
                # x = quantized_perturbe[0].detach() + quantized[1].detach() + quantized[2]
        else:
            x = quantized[0].detach() + quantized[1].detach()
            # x = quantized_perturbe[0].detach() + quantized[1].detach()

        if self.use_gr_x_timbre:
            (x_timbre,) = self.x_timbre_predictor(x)
            out["x_timbre"] = x_timbre

        x = x.transpose(1, 2)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2)
        x = x * gamma + beta

        x = self.model(x)
        out["audio"] = x

        return out

    def vq2emb(self, vq, use_residual_code=True):
        # vq: [num_quantizer, B, T]
        self.quantizer = self.quantizer.eval()
        out = 0
        out += self.quantizer[0].vq2emb(vq[0 : self.vq_num_q_p])
        out += self.quantizer[1].vq2emb(
            vq[self.vq_num_q_p : self.vq_num_q_p + self.vq_num_q_c]
        )
        if self.vq_num_q_r > 0 and use_residual_code:
            out += self.quantizer[2].vq2emb(vq[self.vq_num_q_p + self.vq_num_q_c :])
        return out

    def inference(self, x, speaker_embedding):
        style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        x = x.transpose(1, 2)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2)
        x = x * gamma + beta
        x = self.model(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class FACodecRedecoder(nn.Module):
    def __init__(
        self,
        in_channels=256,
        upsample_initial_channel=1280,
        up_ratios=(5, 5, 4, 2),
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.up_ratios = up_ratios

        self.vq_num_q_p = vq_num_q_p
        self.vq_num_q_c = vq_num_q_c
        self.vq_num_q_r = vq_num_q_r

        self.vq_dim = vq_dim

        self.codebook_size_prosody = codebook_size_prosody
        self.codebook_size_content = codebook_size_content
        self.codebook_size_residual = codebook_size_residual

        self.prosody_embs = nn.ModuleList()
        for i in range(self.vq_num_q_p):
            emb_tokens = nn.Embedding(
                num_embeddings=2**self.codebook_size_prosody,
                embedding_dim=self.vq_dim,
            )
            emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)
            self.prosody_embs.append(emb_tokens)
        self.content_embs = nn.ModuleList()
        for i in range(self.vq_num_q_c):
            emb_tokens = nn.Embedding(
                num_embeddings=2**self.codebook_size_content,
                embedding_dim=self.vq_dim,
            )
            emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)
            self.content_embs.append(emb_tokens)
        self.residual_embs = nn.ModuleList()
        for i in range(self.vq_num_q_r):
            emb_tokens = nn.Embedding(
                num_embeddings=2**self.codebook_size_residual,
                embedding_dim=self.vq_dim,
            )
            emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)
            self.residual_embs.append(emb_tokens)

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
            Activation1d(activation=SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.timbre_cond_prosody_enc = TransformerEncoder(
            enc_emb_tokens=None,
            encoder_layer=4,
            encoder_hidden=256,
            encoder_head=4,
            conv_filter_size=1024,
            conv_kernel_size=5,
            encoder_dropout=0.1,
            use_cln=True,
            cfg=None,
        )

    def forward(
        self,
        vq,
        speaker_embedding,
        use_residual_code=False,
    ):

        x = 0

        x_p = 0
        for i in range(self.vq_num_q_p):
            x_p = x_p + self.prosody_embs[i](vq[i])  # (B, T, d)
        spk_cond = speaker_embedding.unsqueeze(1).expand(-1, x_p.shape[1], -1)
        x_p = self.timbre_cond_prosody_enc(
            x_p, key_padding_mask=None, condition=spk_cond
        )
        x = x + x_p

        x_c = 0
        for i in range(self.vq_num_q_c):
            x_c = x_c + self.content_embs[i](vq[self.vq_num_q_p + i])

        x = x + x_c

        if use_residual_code:

            x_r = 0
            for i in range(self.vq_num_q_r):
                x_r = x_r + self.residual_embs[i](
                    vq[self.vq_num_q_p + self.vq_num_q_c + i]
                )
            x = x + x_r

        style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        x = x.transpose(1, 2)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2)
        x = x * gamma + beta
        x = self.model(x)

        return x

    def vq2emb(self, vq, speaker_embedding, use_residual=True):

        out = 0

        x_t = 0
        for i in range(self.vq_num_q_p):
            x_t += self.prosody_embs[i](vq[i])  # (B, T, d)
            spk_cond = speaker_embedding.unsqueeze(1).expand(-1, x_t.shape[1], -1)
            x_t = self.timbre_cond_prosody_enc(
                x_t, key_padding_mask=None, condition=spk_cond
            )

        # prosody
        out += x_t

        # content
        for i in range(self.vq_num_q_c):
            out += self.content_embs[i](vq[self.vq_num_q_p + i])

        # residual
        if use_residual:
            for i in range(self.vq_num_q_r):
                out += self.residual_embs[i](vq[self.vq_num_q_p + self.vq_num_q_c + i])

        out = out.transpose(1, 2)  # (B, T, d) -> (B, d, T)
        return out

    def inference(self, x, speaker_embedding):
        style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        x = x.transpose(1, 2)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2)
        x = x * gamma + beta
        x = self.model(x)
        return x


class FACodecEncoderV2(nn.Module):
    def __init__(
        self,
        ngf=32,
        up_ratios=(2, 4, 5, 5),
        out_channels=1024,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.up_ratios = up_ratios

        # Create first convolution
        d_model = ngf
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in up_ratios:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Activation1d(activation=SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

        self.mel_transform = MelSpectrogram(
            n_fft=1024,
            num_mels=80,
            sampling_rate=16000,
            hop_size=200,
            win_size=800,
            fmin=0,
            fmax=8000,
        )

        self.reset_parameters()

    def forward(self, x):
        out = self.block(x)
        return out

    def inference(self, x):
        return self.block(x)

    def get_prosody_feature(self, x):
        return self.mel_transform(x.squeeze(1))[:, :20, :]

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)


class FACodecDecoderV2(nn.Module):
    def __init__(
        self,
        in_channels=256,
        upsample_initial_channel=1536,
        ngf=32,
        up_ratios=(5, 5, 4, 2),
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=1024,
        vq_commit_weight=0.005,
        vq_weight_init=False,
        vq_full_commit_loss=False,
        codebook_dim=8,
        codebook_size_prosody=10,  # true codebook size is equal to 2^codebook_size
        codebook_size_content=10,
        codebook_size_residual=10,
        quantizer_dropout=0.0,
        dropout_type="linear",
        use_gr_content_f0=False,
        use_gr_prosody_phone=False,
        use_gr_residual_f0=False,
        use_gr_residual_phone=False,
        use_gr_x_timbre=False,
        use_random_mask_residual=True,
        prob_random_mask_residual=0.75,
    ):
        super().__init__()
        self.hop_length = np.prod(up_ratios)
        self.ngf = ngf
        self.up_ratios = up_ratios

        self.use_random_mask_residual = use_random_mask_residual
        self.prob_random_mask_residual = prob_random_mask_residual

        self.vq_num_q_p = vq_num_q_p
        self.vq_num_q_c = vq_num_q_c
        self.vq_num_q_r = vq_num_q_r

        self.codebook_size_prosody = codebook_size_prosody
        self.codebook_size_content = codebook_size_content
        self.codebook_size_residual = codebook_size_residual

        quantizer_class = ResidualVQ

        self.quantizer = nn.ModuleList()

        # prosody
        quantizer = quantizer_class(
            num_quantizers=vq_num_q_p,
            dim=vq_dim,
            codebook_size=codebook_size_prosody,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
            quantizer_dropout=quantizer_dropout,
            dropout_type=dropout_type,
        )
        self.quantizer.append(quantizer)

        # phone
        quantizer = quantizer_class(
            num_quantizers=vq_num_q_c,
            dim=vq_dim,
            codebook_size=codebook_size_content,
            codebook_dim=codebook_dim,
            threshold_ema_dead_code=2,
            commitment=vq_commit_weight,
            weight_init=vq_weight_init,
            full_commit_loss=vq_full_commit_loss,
            quantizer_dropout=quantizer_dropout,
            dropout_type=dropout_type,
        )
        self.quantizer.append(quantizer)

        # residual
        if self.vq_num_q_r > 0:
            quantizer = quantizer_class(
                num_quantizers=vq_num_q_r,
                dim=vq_dim,
                codebook_size=codebook_size_residual,
                codebook_dim=codebook_dim,
                threshold_ema_dead_code=2,
                commitment=vq_commit_weight,
                weight_init=vq_weight_init,
                full_commit_loss=vq_full_commit_loss,
                quantizer_dropout=quantizer_dropout,
                dropout_type=dropout_type,
            )
            self.quantizer.append(quantizer)

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
            Activation1d(activation=SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.timbre_encoder = TransformerEncoder(
            enc_emb_tokens=None,
            encoder_layer=4,
            encoder_hidden=256,
            encoder_head=4,
            conv_filter_size=1024,
            conv_kernel_size=5,
            encoder_dropout=0.1,
            use_cln=False,
        )

        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.f0_predictor = CNNLSTM(in_channels, 1, 2)
        self.phone_predictor = CNNLSTM(in_channels, 5003, 1)

        self.use_gr_content_f0 = use_gr_content_f0
        self.use_gr_prosody_phone = use_gr_prosody_phone
        self.use_gr_residual_f0 = use_gr_residual_f0
        self.use_gr_residual_phone = use_gr_residual_phone
        self.use_gr_x_timbre = use_gr_x_timbre

        if self.vq_num_q_r > 0 and self.use_gr_residual_f0:
            self.res_f0_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 1, 2)
            )

        if self.vq_num_q_r > 0 and self.use_gr_residual_phone > 0:
            self.res_phone_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 5003, 1)
            )

        if self.use_gr_content_f0:
            self.content_f0_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 1, 2)
            )

        if self.use_gr_prosody_phone:
            self.prosody_phone_predictor = nn.Sequential(
                GradientReversal(alpha=1.0), CNNLSTM(in_channels, 5003, 1)
            )

        if self.use_gr_x_timbre:
            self.x_timbre_predictor = nn.Sequential(
                GradientReversal(alpha=1),
                CNNLSTM(in_channels, 245200, 1, global_pred=True),
            )

        self.melspec_linear = nn.Linear(20, 256)
        self.melspec_encoder = TransformerEncoder(
            enc_emb_tokens=None,
            encoder_layer=4,
            encoder_hidden=256,
            encoder_head=4,
            conv_filter_size=1024,
            conv_kernel_size=5,
            encoder_dropout=0.1,
            use_cln=False,
            cfg=None,
        )

        self.reset_parameters()

    def quantize(self, x, prosody_feature, n_quantizers=None):
        outs, qs, commit_loss, quantized_buf = 0, [], [], []

        # prosody
        f0_input = prosody_feature.transpose(1, 2)  # (B, T, 20)
        f0_input = self.melspec_linear(f0_input)
        f0_input = self.melspec_encoder(f0_input, None, None)
        f0_input = f0_input.transpose(1, 2)
        f0_quantizer = self.quantizer[0]
        out, q, commit, quantized = f0_quantizer(f0_input, n_quantizers=n_quantizers)
        outs += out
        qs.append(q)
        quantized_buf.append(quantized.sum(0))
        commit_loss.append(commit)

        # phone
        phone_input = x
        phone_quantizer = self.quantizer[1]
        out, q, commit, quantized = phone_quantizer(
            phone_input, n_quantizers=n_quantizers
        )
        outs += out
        qs.append(q)
        quantized_buf.append(quantized.sum(0))
        commit_loss.append(commit)

        # residual
        if self.vq_num_q_r > 0:
            residual_quantizer = self.quantizer[2]
            residual_input = x - (quantized_buf[0] + quantized_buf[1]).detach()
            out, q, commit, quantized = residual_quantizer(
                residual_input, n_quantizers=n_quantizers
            )
            outs += out
            qs.append(q)
            quantized_buf.append(quantized.sum(0))  # [L, B, C, T] -> [B, C, T]
            commit_loss.append(commit)

        qs = torch.cat(qs, dim=0)
        commit_loss = torch.cat(commit_loss, dim=0)
        return outs, qs, commit_loss, quantized_buf

    def forward(
        self,
        x,
        prosody_feature,
        vq=True,
        get_vq=False,
        eval_vq=True,
        speaker_embedding=None,
        n_quantizers=None,
        quantized=None,
    ):
        if get_vq:
            return self.quantizer.get_emb()
        if vq is True:
            if eval_vq:
                self.quantizer.eval()
            x_timbre = x
            outs, qs, commit_loss, quantized_buf = self.quantize(
                x, prosody_feature, n_quantizers=n_quantizers
            )

            x_timbre = x_timbre.transpose(1, 2)
            x_timbre = self.timbre_encoder(x_timbre, None, None)
            x_timbre = x_timbre.transpose(1, 2)
            spk_embs = torch.mean(x_timbre, dim=2)
            return outs, qs, commit_loss, quantized_buf, spk_embs

        out = {}

        layer_0 = quantized[0]
        f0, uv = self.f0_predictor(layer_0)
        f0 = rearrange(f0, "... 1 -> ...")
        uv = rearrange(uv, "... 1 -> ...")

        layer_1 = quantized[1]
        (phone,) = self.phone_predictor(layer_1)

        out = {"f0": f0, "uv": uv, "phone": phone}

        if self.use_gr_prosody_phone:
            (prosody_phone,) = self.prosody_phone_predictor(layer_0)
            out["prosody_phone"] = prosody_phone

        if self.use_gr_content_f0:
            content_f0, content_uv = self.content_f0_predictor(layer_1)
            content_f0 = rearrange(content_f0, "... 1 -> ...")
            content_uv = rearrange(content_uv, "... 1 -> ...")
            out["content_f0"] = content_f0
            out["content_uv"] = content_uv

        if self.vq_num_q_r > 0:
            layer_2 = quantized[2]

            if self.use_gr_residual_f0:
                res_f0, res_uv = self.res_f0_predictor(layer_2)
                res_f0 = rearrange(res_f0, "... 1 -> ...")
                res_uv = rearrange(res_uv, "... 1 -> ...")
                out["res_f0"] = res_f0
                out["res_uv"] = res_uv

            if self.use_gr_residual_phone:
                (res_phone,) = self.res_phone_predictor(layer_2)
                out["res_phone"] = res_phone

        style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if self.vq_num_q_r > 0:
            if self.use_random_mask_residual:
                bsz = quantized[2].shape[0]
                res_mask = np.random.choice(
                    [0, 1],
                    size=bsz,
                    p=[
                        self.prob_random_mask_residual,
                        1 - self.prob_random_mask_residual,
                    ],
                )
                res_mask = (
                    torch.from_numpy(res_mask).unsqueeze(1).unsqueeze(1)
                )  # (B, 1, 1)
                res_mask = res_mask.to(
                    device=quantized[2].device, dtype=quantized[2].dtype
                )
                x = (
                    quantized[0].detach()
                    + quantized[1].detach()
                    + quantized[2] * res_mask
                )
                # x = quantized_perturbe[0].detach() + quantized[1].detach() + quantized[2] * res_mask
            else:
                x = quantized[0].detach() + quantized[1].detach() + quantized[2]
                # x = quantized_perturbe[0].detach() + quantized[1].detach() + quantized[2]
        else:
            x = quantized[0].detach() + quantized[1].detach()
            # x = quantized_perturbe[0].detach() + quantized[1].detach()

        if self.use_gr_x_timbre:
            (x_timbre,) = self.x_timbre_predictor(x)
            out["x_timbre"] = x_timbre

        x = x.transpose(1, 2)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2)
        x = x * gamma + beta

        x = self.model(x)
        out["audio"] = x

        return out

    def vq2emb(self, vq, use_residual=True):
        # vq: [num_quantizer, B, T]
        self.quantizer = self.quantizer.eval()
        out = 0
        out += self.quantizer[0].vq2emb(vq[0 : self.vq_num_q_p])
        out += self.quantizer[1].vq2emb(
            vq[self.vq_num_q_p : self.vq_num_q_p + self.vq_num_q_c]
        )
        if self.vq_num_q_r > 0 and use_residual:
            out += self.quantizer[2].vq2emb(vq[self.vq_num_q_p + self.vq_num_q_c :])
        return out

    def inference(self, x, speaker_embedding):
        style = self.timbre_linear(speaker_embedding).unsqueeze(2)  # (B, 2d, 1)
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        x = x.transpose(1, 2)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2)
        x = x * gamma + beta
        x = self.model(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        self.apply(init_weights)
