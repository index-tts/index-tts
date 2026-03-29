from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, MutableMapping, Sequence, TypeVar

import torch


@dataclass(frozen=True)
class WeightedReference:
    path: str
    weight: float


T = TypeVar("T")


def normalize_reference_inputs(reference_paths, weights=None, label="references"):
    refs = _coerce_reference_paths(reference_paths, label)
    normalized_weights = _normalize_weights(weights, len(refs), label)
    return [
        WeightedReference(path=path, weight=weight)
        for path, weight in zip(refs, normalized_weights)
    ]


def merge_weighted_vectors(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Expected at least one tensor to merge")
    if len(tensors) != len(weights):
        raise ValueError("Tensor and weight counts must match")
    if len(tensors) == 1:
        return tensors[0]

    ref_tensor = tensors[0]
    weight_tensor = torch.tensor(weights, device=ref_tensor.device, dtype=ref_tensor.dtype)
    stacked = torch.stack([tensor.to(device=ref_tensor.device, dtype=ref_tensor.dtype) for tensor in tensors], dim=0)
    weight_shape = [len(weights)] + [1] * (stacked.ndim - 1)
    return (stacked * weight_tensor.view(*weight_shape)).sum(dim=0)


def merge_variable_length_tensors(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    seq_dim: int = -1,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("Expected at least one tensor to merge")
    if len(tensors) != len(weights):
        raise ValueError("Tensor and weight counts must match")
    if len(tensors) == 1:
        return tensors[0]

    normalized_tensors = [tensor.movedim(seq_dim, -1) for tensor in tensors]
    ref_non_sequence_shape = normalized_tensors[0].shape[:-1]
    for index, tensor in enumerate(normalized_tensors[1:], start=1):
        if tensor.shape[:-1] != ref_non_sequence_shape:
            raise ValueError(
                "All tensor dimensions except the sequence dimension must match for merging. "
                f"Expected {ref_non_sequence_shape} before the sequence axis, got {tensor.shape[:-1]} "
                f"for tensor index {index}. Check the seq_dim argument."
            )
    lengths = torch.tensor(
        [tensor.shape[-1] for tensor in normalized_tensors],
        device=normalized_tensors[0].device,
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())
    ref_tensor = normalized_tensors[0]
    padded = []
    for tensor in normalized_tensors:
        pad_shape = list(tensor.shape)
        pad_shape[-1] = max_len
        padded_tensor = tensor.new_zeros(pad_shape)
        padded_tensor[..., : tensor.shape[-1]] = tensor
        padded.append(padded_tensor)

    stacked = torch.stack(padded, dim=0)
    weight_tensor = torch.tensor(weights, device=ref_tensor.device, dtype=ref_tensor.dtype)
    frame_index = torch.arange(max_len, device=ref_tensor.device)
    active_mask = frame_index.unsqueeze(0) < lengths.unsqueeze(1)
    weighted_mask = weight_tensor.unsqueeze(1) * active_mask.to(weight_tensor.dtype)
    denominator = weighted_mask.sum(dim=0).clamp_min(torch.finfo(weight_tensor.dtype).eps)
    normalized_mask = weighted_mask / denominator.unsqueeze(0)
    weight_shape = [len(tensors)] + [1] * (stacked.ndim - 2) + [max_len]
    merged = (stacked * normalized_mask.view(*weight_shape)).sum(dim=0)
    return merged.movedim(-1, seq_dim)


def get_or_create_cached(cache: MutableMapping[str, T], key: str, factory: Callable[[], T]) -> T:
    cached = cache.get(key)
    if cached is not None:
        return cached

    cache[key] = factory()
    return cache[key]


def _coerce_reference_paths(reference_paths, label: str) -> list[str]:
    if isinstance(reference_paths, (str, os.PathLike)):
        raw_paths = [reference_paths]
    elif isinstance(reference_paths, Sequence):
        raw_paths = list(reference_paths)
    else:
        raise TypeError(f"{label} must be a path or a sequence of paths")

    paths = []
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        path = os.fspath(raw_path).strip()
        if path:
            paths.append(os.path.abspath(path))

    if not paths:
        raise ValueError(f"At least one {label} entry is required")
    return paths


def _normalize_weights(weights, count: int, label: str) -> list[float]:
    if weights is None:
        return [1.0 / count] * count

    if isinstance(weights, (int, float)):
        raw_weights = [float(weights)]
    elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        raw_weights = [float(weight) for weight in weights]
    else:
        raise TypeError(f"{label} weights must be numeric or a sequence of numerics")

    if len(raw_weights) != count:
        raise ValueError(f"{label} weights must match the number of references")
    if any(weight < 0 for weight in raw_weights):
        raise ValueError(f"{label} weights must be non-negative")

    weight_sum = sum(raw_weights)
    if weight_sum <= 0:
        raise ValueError(f"{label} weights must sum to a value greater than 0")

    return [weight / weight_sum for weight in raw_weights]
