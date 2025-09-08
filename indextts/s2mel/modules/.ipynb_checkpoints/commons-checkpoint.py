import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from munch import Munch
import json
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

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


def avg_with_mask(x, mask):
    assert mask.dtype == torch.float, "Mask should be float"

    if mask.ndim == 2:
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1:
        mask = mask.expand_as(x)

    return (x * mask).sum() / mask.sum()


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


def load_F0_models(path):
    # load F0 model
    from .JDC.model import JDCNet

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location="cpu")["net"]
    F0_model.load_state_dict(params)
    _ = F0_model.train()

    return F0_model


def modify_w2v_forward(self, output_layer=15):
    """
    change forward method of w2v encoder to get its intermediate layer output
    :param self:
    :param layer:
    :return:
    """
    from transformers.modeling_outputs import BaseModelOutput

    def forward(
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(
                ~attention_mask.bool().unsqueeze(-1), 0.0
            )

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(
                dtype=hidden_states.dtype
            )
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        deepspeed_zero3_is_enabled = False

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = (
                True
                if self.training and (dropout_probability < self.config.layerdrop)
                else False
            )
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        relative_position_embeddings,
                        output_attentions,
                        conv_attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if i == output_layer - 1:
                break

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    return forward


MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        import logging

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def normalize_f0(f0_sequence):
    # Remove unvoiced frames (replace with -1)
    voiced_indices = np.where(f0_sequence > 0)[0]
    f0_voiced = f0_sequence[voiced_indices]

    # Convert to log scale
    log_f0 = np.log2(f0_voiced)

    # Calculate mean and standard deviation
    mean_f0 = np.mean(log_f0)
    std_f0 = np.std(log_f0)

    # Normalize the F0 sequence
    normalized_f0 = (log_f0 - mean_f0) / std_f0

    # Create the normalized F0 sequence with unvoiced frames
    normalized_sequence = np.zeros_like(f0_sequence)
    normalized_sequence[voiced_indices] = normalized_f0
    normalized_sequence[f0_sequence <= 0] = -1  # Assign -1 to unvoiced frames

    return normalized_sequence


class MyModel(nn.Module):
    def __init__(self,args):
        super(MyModel, self).__init__()
        from modules.flow_matching import CFM
        from modules.length_regulator import InterpolateRegulator
        
        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels if hasattr(args.length_regulator, "in_channels") else None,
            vector_quantize=args.length_regulator.vector_quantize if hasattr(args.length_regulator, "vector_quantize") else False,
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=args.length_regulator.n_codebooks if hasattr(args.length_regulator, "n_codebooks") else 1,
            quantizer_dropout=args.length_regulator.quantizer_dropout if hasattr(args.length_regulator, "quantizer_dropout") else 0.0,
            f0_condition=args.length_regulator.f0_condition if hasattr(args.length_regulator, "f0_condition") else False,
            n_f0_bins=args.length_regulator.n_f0_bins if hasattr(args.length_regulator, "n_f0_bins") else 512,
        )

        self.models = nn.ModuleDict({
            'cfm': CFM(args),
            'length_regulator': length_regulator 
        })
    
    def forward(self, x, target_lengths, prompt_len, cond, y):
        x = self.models['cfm'](x, target_lengths, prompt_len, cond, y)
        return x
    
    def forward2(self, S_ori,target_lengths,F0_ori):
        x = self.models['length_regulator'](S_ori, ylens=target_lengths, f0=F0_ori)
        return x

def build_model(args, stage="DiT"):
    if stage == "DiT":
        from modules.flow_matching import CFM
        from modules.length_regulator import InterpolateRegulator
        
        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels if hasattr(args.length_regulator, "in_channels") else None,
            vector_quantize=args.length_regulator.vector_quantize if hasattr(args.length_regulator, "vector_quantize") else False,
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=args.length_regulator.n_codebooks if hasattr(args.length_regulator, "n_codebooks") else 1,
            quantizer_dropout=args.length_regulator.quantizer_dropout if hasattr(args.length_regulator, "quantizer_dropout") else 0.0,
            f0_condition=args.length_regulator.f0_condition if hasattr(args.length_regulator, "f0_condition") else False,
            n_f0_bins=args.length_regulator.n_f0_bins if hasattr(args.length_regulator, "n_f0_bins") else 512,
        )
        cfm = CFM(args)
        nets = Munch(
            cfm=cfm,
            length_regulator=length_regulator,
        )
        
    elif stage == 'codec':
        from dac.model.dac import Encoder
        from modules.quantize import (
            FAquantizer,
        )

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

        nets = Munch(
            encoder=encoder,
            quantizer=quantizer,
        )

    elif stage == "mel_vocos":
        from modules.vocos import Vocos
        decoder = Vocos(args)
        nets = Munch(
            decoder=decoder,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return nets


def load_checkpoint(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
    load_ema=False,
):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        print("Loading EMA")
        for key in model:
            i = 0
            for param_name in params[key]:
                if "input_pos" in param_name:
                    continue
                assert params[key][param_name].shape == state["ema"][key][0][i].shape
                params[key][param_name] = state["ema"][key][0][i].clone()
                i += 1
    for key in model:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                # strip prefix of DDP (module.), create a new OrderedDict that does not contain the prefix
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            model_state_dict = model[key].state_dict()
            # 过滤出形状匹配的键值对
            filtered_state_dict = {
                k: v
                for k, v in params[key].items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                print(
                    f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
                )
            print("%s loaded" % key)
            model[key].load_state_dict(filtered_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])

    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters

def load_checkpoint2(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
    load_ema=False,
):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        print("Loading EMA")
        for key in model.models:
            i = 0
            for param_name in params[key]:
                if "input_pos" in param_name:
                    continue
                assert params[key][param_name].shape == state["ema"][key][0][i].shape
                params[key][param_name] = state["ema"][key][0][i].clone()
                i += 1
    for key in model.models:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                # strip prefix of DDP (module.), create a new OrderedDict that does not contain the prefix
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            model_state_dict = model.models[key].state_dict()
            # 过滤出形状匹配的键值对
            filtered_state_dict = {
                k: v
                for k, v in params[key].items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                print(
                    f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
                )
            print("%s loaded" % key)
            model.models[key].load_state_dict(filtered_state_dict, strict=False)
    model.eval()
#     _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])

    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
