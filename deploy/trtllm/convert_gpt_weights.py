"""Weight conversion utilities for IndexTTS2 GPT to TRT-LLM checkpoint."""

import os
import time
from typing import Dict, Optional

import torch
from omegaconf import OmegaConf
from transformers import GPT2Config

from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_torch
from tensorrt_llm.models.convert_utils import get_weight, get_weight_and_bias
from tensorrt_llm.models.gpt.config import GPTConfig
from tensorrt_llm.quantization import QuantAlgo

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint


def split(param: torch.Tensor,
          tp_rank: int,
          tp_size: int,
          is_column: bool = True) -> torch.Tensor:
    """Split linear layer's weight, bias or scaling factors for tensor parallelism."""
    if param is None:
        return None
    assert param.ndim in [1, 2]
    if tp_size == 1:
        return param
    if param.numel() == 1:
        return param
    if param.ndim == 1 and not is_column:
        return param
    split_dim = 0 if (param.ndim == 1 or is_column) else 1
    return torch.chunk(param, tp_size, dim=split_dim)[tp_rank].contiguous()


def split_qkv(
    param: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """Split qkv layer's weight, bias or scaling factors for tensor parallelism."""
    if param is None:
        return None
    assert hidden_size % num_heads == 0
    head_dim = hidden_size // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    assert num_heads % num_kv_heads == 0
    assert num_heads % tp_size == 0

    q_param, k_param, v_param = torch.split(
        param, [hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim],
        dim=0)

    if num_kv_heads < tp_size:
        assert tp_size % num_kv_heads == 0
        num_dups = tp_size // num_kv_heads
        remain_shape = k_param.shape[1:]
        k_param = k_param.view(
            num_kv_heads, head_dim,
            *remain_shape).repeat_interleave(num_dups, dim=0).view(
                num_kv_heads * head_dim * num_dups, *remain_shape)
        v_param = v_param.view(
            num_kv_heads, head_dim,
            *remain_shape).repeat_interleave(num_dups, dim=0).view(
                num_kv_heads * head_dim * num_dups, *remain_shape)
    else:
        assert num_kv_heads % tp_size == 0

    q_param = split(q_param, tp_rank, tp_size, is_column=True)
    k_param = split(k_param, tp_rank, tp_size, is_column=True)
    v_param = split(v_param, tp_rank, tp_size, is_column=True)
    return torch.cat([q_param, k_param, v_param], dim=0)


def split_embedding(
    param: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    use_parallel_embedding: bool = False,
    sharding_dim: int = 0,
) -> torch.Tensor:
    if param is None:
        return None
    if not use_parallel_embedding:
        return param

    vocab_size, hidden_size = param.size()
    if sharding_dim == 0:
        if vocab_size % tp_size != 0:
            vocab_size_padded = pad_vocab_size(vocab_size, tp_size)
            pad_width = vocab_size_padded - vocab_size
            param = torch.nn.functional.pad(param, (0, 0, 0, pad_width),
                                            value=0)
        else:
            assert hidden_size % tp_size == 0
    return split(param, tp_rank, tp_size, is_column=(sharding_dim == 0))


def get_tllm_linear_weight(
    weight: torch.Tensor,
    prefix: str,
    bias: Optional[torch.Tensor] = None,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8
) -> Dict[str, torch.Tensor]:
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[f'{prefix}.weight'] = processed_torch_weights
        results[f'{prefix}.per_channel_scale'] = torch_weight_scales
    else:
        results[f'{prefix}.weight'] = weight

    if bias is not None:
        results[f'{prefix}.bias'] = bias

    return results


def load_weights_from_hf_model(hf_weights,
                               hf_config,
                               config: GPTConfig,
                               act_range: Optional[dict] = None):
    """Convert IndexTTS2 GPT weights to TRT-LLM format.

    Supports FP16 and weight-only quantization (W8A16, W4A16).
    """
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    weights = {}
    tik = time.time()

    model_params = hf_weights

    dtype = getattr(torch, config.dtype)
    gpt_variant = config.gpt_variant
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    num_kv_heads = config.num_key_value_heads
    num_hidden_layers = config.num_hidden_layers
    multi_query_mode = (num_kv_heads != num_attention_heads)

    mapping = config.mapping
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'transformer.h.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        # (1) Attention QKV Linear
        qkv_w, qkv_b = get_weight_and_bias(model_params, f'{prefix}.attn.c_attn', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            qkv_w = qkv_w.t().contiguous()

        qkv_w = split_qkv(qkv_w, mapping.tp_rank, mapping.tp_size,
                          hidden_size, num_attention_heads, num_kv_heads)
        qkv_b = split_qkv(qkv_b, mapping.tp_rank, mapping.tp_size,
                          hidden_size, num_attention_heads, num_kv_heads)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv',
                                   qkv_b, use_weight_only,
                                   plugin_weight_only_quant_type))

        # (2) Attention Dense Linear
        attn_dense_w, attn_dense_b = get_weight_and_bias(model_params, f'{prefix}.attn.c_proj', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            attn_dense_w = attn_dense_w.t().contiguous()

        attn_dense_w = split(attn_dense_w, mapping.tp_rank, mapping.tp_size, is_column=False)
        weights.update(
            get_tllm_linear_weight(attn_dense_w, f'{tllm_prex}.attention.dense',
                                   attn_dense_b, use_weight_only,
                                   plugin_weight_only_quant_type))

        # (3) MLP FC Linear
        mlp_fc_w, mlp_fc_b = get_weight_and_bias(model_params, f'{prefix}.mlp.c_fc', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            mlp_fc_w = mlp_fc_w.t().contiguous()

        mlp_fc_w = split(mlp_fc_w, mapping.tp_rank, mapping.tp_size, is_column=True)
        mlp_fc_b = split(mlp_fc_b, mapping.tp_rank, mapping.tp_size, is_column=True)
        weights.update(
            get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc',
                                   mlp_fc_b, use_weight_only,
                                   plugin_weight_only_quant_type))

        # (4) MLP Proj Layer
        mlp_proj_w, mlp_proj_b = get_weight_and_bias(model_params, f'{prefix}.mlp.c_proj', dtype)
        if gpt_variant in ['gpt2', 'santacoder', 'jais']:
            mlp_proj_w = mlp_proj_w.t().contiguous()

        mlp_proj_w = split(mlp_proj_w, mapping.tp_rank, mapping.tp_size, is_column=False)
        weights.update(
            get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                   mlp_proj_b, use_weight_only,
                                   plugin_weight_only_quant_type))

        # (5) Input layernorm
        input_ln_w, input_ln_b = get_weight_and_bias(model_params, f'{prefix}.ln_1', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_w
        if input_ln_b is not None:
            weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_b

        # (6) Post layernorm
        post_ln_w, post_ln_b = get_weight_and_bias(model_params, f'{prefix}.ln_2', dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_w
        if post_ln_b is not None:
            weights[f'{tllm_prex}.post_layernorm.bias'] = post_ln_b

    if mapping.is_first_pp_rank():
        embed_w = get_weight(model_params, 'transformer.wte', dtype)
        weights['transformer.vocab_embedding.weight'] = split_embedding(
            embed_w, mapping.tp_rank, mapping.tp_size,
            use_parallel_embedding=config.use_parallel_embedding,
            sharding_dim=config.embedding_sharding_dim)

        pos_embed_w = get_weight(model_params, 'transformer.wpe', dtype)
        if pos_embed_w is not None:
            weights['transformer.position_embedding.weight'] = split_embedding(
                pos_embed_w, mapping.tp_rank, mapping.tp_size,
                use_parallel_embedding=config.use_parallel_embedding,
                sharding_dim=config.embedding_sharding_dim)

    if mapping.is_last_pp_rank():
        lm_head_w, lm_head_b = get_weight_and_bias(model_params, 'lm_head', dtype)
        output_vocab_size = lm_head_w.shape[0]
        if output_vocab_size % mapping.tp_size != 0:
            output_vocab_size_padded = pad_vocab_size(output_vocab_size, mapping.tp_size)
            pad_width = output_vocab_size_padded - output_vocab_size
            lm_head_w = torch.nn.functional.pad(lm_head_w, (0, 0, 0, pad_width), value=0)
            lm_head_b = torch.nn.functional.pad(lm_head_b, (0, pad_width), value=0)

        if hasattr(hf_config, 'logits_scale'):
            lm_head_w *= hf_config.logits_scale

        weights['lm_head.weight'] = split(lm_head_w.clone(), mapping.tp_rank, mapping.tp_size, is_column=True)
        weights['lm_head.bias'] = split(lm_head_b.clone(), mapping.tp_rank, mapping.tp_size, is_column=True)

        ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def load_hf_weights_and_config(cfg_path: str, model_dir: str, load_model_on_cpu: bool = False):
    """Load IndexTTS2 GPT weights and build a GPT2Config for TRT-LLM conversion.

    Handles the custom embedding layout:
    - vocab_embedding = cat(mel_embedding, text_embedding)
    - position_embedding = cat(mel_pos_embedding, text_pos_embedding, zero_padding)
    """
    cfg = OmegaConf.load(cfg_path)

    gpt = UnifiedVoice(**cfg.gpt, use_accel=False)
    gpt_path = os.path.join(model_dir, cfg.gpt_checkpoint)
    load_checkpoint(gpt, gpt_path)

    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
    hf_model = gpt.inference_model

    vocab_size = gpt.number_mel_codes + (gpt.number_text_tokens + 1)

    max_mel_seq_len = gpt.max_mel_tokens + 2 + gpt.max_conditioning_inputs
    max_text_seq_len = gpt.max_text_tokens + 2

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_mel_seq_len + max_text_seq_len + 1,
        n_embd=gpt.model_dim,
        n_layer=gpt.layers,
        n_head=gpt.heads
    )

    config.architectures = ['IndexTTS2GPTForCausalLM']
    config.output_vocab_size = gpt.number_mel_codes

    model_params = dict(hf_model.named_parameters())

    mel_embedding_params = dict(gpt.mel_embedding.named_parameters())
    text_embedding_params = dict(gpt.text_embedding.named_parameters())
    mel_pos_embedding_params = dict(gpt.mel_pos_embedding.named_parameters())
    text_pos_embedding_params = dict(gpt.text_pos_embedding.named_parameters())

    # Combined vocab embedding: [mel_codes | text_tokens]
    model_params["transformer.wte.weight"] = torch.cat(
        [mel_embedding_params["weight"], text_embedding_params["weight"]], dim=0)

    # Combined position embedding: [mel_pos | text_pos | zero_for_ptuning]
    model_params["transformer.wpe.weight"] = torch.cat([
        mel_pos_embedding_params["emb.weight"],
        text_pos_embedding_params["emb.weight"],
        torch.zeros((1, mel_pos_embedding_params["emb.weight"].shape[1]),
                    device=mel_pos_embedding_params["emb.weight"].device)
    ], dim=0)

    # Rename lm_head.1 -> lm_head
    model_params["lm_head.weight"] = model_params.pop("lm_head.1.weight")
    model_params["lm_head.bias"] = model_params.pop("lm_head.1.bias")

    # Rename final_norm -> transformer.ln_f
    model_params["transformer.ln_f.weight"] = model_params.pop("final_norm.weight")
    model_params["transformer.ln_f.bias"] = model_params.pop("final_norm.bias")

    # Remove unused
    del model_params["text_pos_embedding.emb.weight"]

    return model_params, config
