# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import os.path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from munch import Munch
import json


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments_audio(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(
        dtype=torch.long
    )
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


from huggingface_hub import hf_hub_download


def load_F0_models(path):
    # load F0 model
    from .JDC.model import JDCNet

    F0_model = JDCNet(num_class=1, seq_len=192)
    if not os.path.exists(path):
        path = hf_hub_download(repo_id="Plachta/JDCnet", filename="bst.t7")
    params = torch.load(path, map_location="cpu")["net"]
    F0_model.load_state_dict(params)
    _ = F0_model.train()

    return F0_model


# Generators
from modules.dac.model.dac import Encoder, Decoder
from .quantize import FAquantizer, FApredictors

# Discriminators
from modules.dac.model.discriminator import Discriminator


def build_model(args):
    encoder = Encoder(
        d_model=args.DAC.encoder_dim,
        strides=args.DAC.encoder_rates,
        d_latent=1024,
        causal=args.causal,
        lstm=args.lstm,
    )

    quantizer = FAquantizer(
        in_dim=1024,
        n_p_codebooks=1,
        n_c_codebooks=args.n_c_codebooks,
        n_t_codebooks=2,
        n_r_codebooks=3,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        causal=args.causal,
        separate_prosody_encoder=args.separate_prosody_encoder,
        timbre_norm=args.timbre_norm,
    )

    fa_predictors = FApredictors(
        in_dim=1024,
        use_gr_content_f0=args.use_gr_content_f0,
        use_gr_prosody_phone=args.use_gr_prosody_phone,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
        use_gr_timbre_content=True,
        use_gr_timbre_prosody=args.use_gr_timbre_prosody,
        use_gr_x_timbre=True,
        norm_f0=args.norm_f0,
        timbre_norm=args.timbre_norm,
        use_gr_content_global_f0=args.use_gr_content_global_f0,
    )

    decoder = Decoder(
        input_channel=1024,
        channels=args.DAC.decoder_dim,
        rates=args.DAC.decoder_rates,
        causal=args.causal,
        lstm=args.lstm,
    )

    discriminator = Discriminator(
        rates=[],
        periods=[2, 3, 5, 7, 11],
        fft_sizes=[2048, 1024, 512],
        sample_rate=args.DAC.sr,
        bands=[(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
    )

    nets = Munch(
        encoder=encoder,
        quantizer=quantizer,
        decoder=decoder,
        discriminator=discriminator,
        fa_predictors=fa_predictors,
    )

    return nets


def load_checkpoint(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    for key in model:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                # strip prefix of DDP (module.), create a new OrderedDict that does not contain the prefix
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            print("%s loaded" % key)
            model[key].load_state_dict(params[key], strict=True)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])

    else:
        epoch = state["epoch"] + 1
        iters = state["iters"]

    return model, optimizer, epoch, iters


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
