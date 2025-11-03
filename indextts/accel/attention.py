from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from torch import nn


@dataclass
class ForwardContext:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_FORWARD_CONTEXT = ForwardContext()


def get_forward_context():
    return _FORWARD_CONTEXT


def set_forward_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _FORWARD_CONTEXT
    _FORWARD_CONTEXT = ForwardContext(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_forward_context():
    global _FORWARD_CONTEXT
    _FORWARD_CONTEXT = ForwardContext()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 2048
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    d_offset = 0
    while d_offset < D:
        cur_block_size = min(BLOCK_SIZE, D - d_offset)
        key_offsets = idx * key_stride + d_offset + tl.arange(0, BLOCK_SIZE)
        value_offsets = idx * value_stride + d_offset + tl.arange(0, BLOCK_SIZE)
        cache_offsets = slot * D + d_offset + tl.arange(0, BLOCK_SIZE)

        mask = tl.arange(0, BLOCK_SIZE) < cur_block_size
        key = tl.load(key_ptr + key_offsets, mask=mask, other=0.0)
        value = tl.load(value_ptr + value_offsets, mask=mask, other=0.0)
        tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
        tl.store(v_cache_ptr + cache_offsets, value, mask=mask)

        d_offset += BLOCK_SIZE


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_forward_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
