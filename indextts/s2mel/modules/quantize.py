from dac.nn.quantize import ResidualVectorQuantize
from torch import nn
from modules.wavenet import WN
import torch
import torchaudio
import torchaudio.functional as audio_F
import numpy as np
from .alias_free_torch import *
from torch.nn.utils import weight_norm
from torch import nn, sin, pow
from einops.layers.torch import Rearrange
from dac.model.encodec import SConv1d

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

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
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

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

def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)
class FAquantizer(nn.Module):
    def __init__(self, in_dim=1024,
                 n_p_codebooks=1,
                 n_c_codebooks=2,
                 n_t_codebooks=2,
                 n_r_codebooks=3,
                 codebook_size=1024,
                 codebook_dim=8,
                 quantizer_dropout=0.5,
                 causal=False,
                 separate_prosody_encoder=False,
                 timbre_norm=False,):
        super(FAquantizer, self).__init__()
        conv1d_type = SConv1d# if causal else nn.Conv1d
        self.prosody_quantizer = ResidualVectorQuantize(
            input_dim=in_dim,
            n_codebooks=n_p_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.content_quantizer = ResidualVectorQuantize(
            input_dim=in_dim,
            n_codebooks=n_c_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.residual_quantizer = ResidualVectorQuantize(
            input_dim=in_dim,
            n_codebooks=n_r_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.melspec_linear = conv1d_type(in_channels=20, out_channels=256, kernel_size=1, causal=causal)
        self.melspec_encoder = WN(hidden_channels=256, kernel_size=5, dilation_rate=1, n_layers=8, gin_channels=0, p_dropout=0.2, causal=causal)
        self.melspec_linear2 = conv1d_type(in_channels=256, out_channels=1024, kernel_size=1, causal=causal)

        self.prob_random_mask_residual = 0.75

        SPECT_PARAMS = {
            "n_fft": 2048,
            "win_length": 1200,
            "hop_length": 300,
        }
        MEL_PARAMS = {
            "n_mels": 80,
        }

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=MEL_PARAMS["n_mels"], sample_rate=24000, **SPECT_PARAMS
        )
        self.mel_mean, self.mel_std = -4, 4
        self.frame_rate = 24000 / 300
        self.hop_length = 300

    def preprocess(self, wave_tensor, n_bins=20):
        mel_tensor = self.to_mel(wave_tensor.squeeze(1))
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mel_mean) / self.mel_std
        return mel_tensor[:, :n_bins, :int(wave_tensor.size(-1) / self.hop_length)]

    def forward(self, x, wave_segments):
        outs = 0
        prosody_feature = self.preprocess(wave_segments)

        f0_input = prosody_feature  # (B, T, 20)
        f0_input = self.melspec_linear(f0_input)
        f0_input = self.melspec_encoder(f0_input, torch.ones(f0_input.shape[0], 1, f0_input.shape[2]).to(
            f0_input.device).bool())
        f0_input = self.melspec_linear2(f0_input)

        common_min_size = min(f0_input.size(2), x.size(2))
        f0_input = f0_input[:, :, :common_min_size]

        x = x[:, :, :common_min_size]

        z_p, codes_p, latents_p, commitment_loss_p, codebook_loss_p = self.prosody_quantizer(
            f0_input, 1
        )
        outs += z_p.detach()

        z_c, codes_c, latents_c, commitment_loss_c, codebook_loss_c = self.content_quantizer(
            x, 2
        )
        outs += z_c.detach()

        residual_feature = x - z_p.detach() - z_c.detach()

        z_r, codes_r, latents_r, commitment_loss_r, codebook_loss_r = self.residual_quantizer(
            residual_feature, 3
        )

        quantized = [z_p, z_c, z_r]
        codes = [codes_p, codes_c, codes_r]

        return quantized, codes