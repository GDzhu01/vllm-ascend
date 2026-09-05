# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Framework-side V4.1 cache specs, hybrid groups and physical allocation."""

from dataclasses import dataclass
from enum import Enum

import torch
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


class CacheGroup(str, Enum):
    SWA = "swa"
    COMPRESSED = "full_ratio2"
    FULL = "full_ratio1"
    STATE = "compressor_state"


@dataclass(frozen=True, kw_only=True)
class DeepseekV41FullSpec(AscendMLAAttentionSpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41FullSpec) and s.compress_ratio == self.compress_ratio for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41SWASpec(AscendSlidingWindowMLASpec):
    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41SWASpec) and s.sliding_window == self.sliding_window for s in specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class DeepseekV41CompressorStateSpec(AscendSlidingWindowMLASpec):
    """V4-style FP32 KV/score rows, retained by SlidingWindowManager.

    State is not compressed: one row per original token, two vectors per row.
    Only pooling has ratio2; the storage compression ratio remains one.
    """

    def is_uniform_with_collection(self, specs):
        return all(
            isinstance(s, DeepseekV41CompressorStateSpec) and s.sliding_window == self.sliding_window
            for s in specs.values()
        )


def is_v41_spec(spec):
    return isinstance(spec, (DeepseekV41FullSpec, DeepseekV41SWASpec, DeepseekV41CompressorStateSpec))


def group_key(spec):
    if isinstance(spec, DeepseekV41SWASpec):
        return CacheGroup.SWA
    if isinstance(spec, DeepseekV41CompressorStateSpec):
        return CacheGroup.STATE
    if isinstance(spec, DeepseekV41FullSpec):
        return CacheGroup.COMPRESSED if spec.compress_ratio == 2 else CacheGroup.FULL
    raise TypeError(f"Not a V4.1 cache spec: {type(spec)}")


def group_cache_specs(specs):
    """Return None for other models, fail closed on mixed unsupported resources."""
    if not any(is_v41_spec(s) for s in specs.values()):
        return None
    if not all(is_v41_spec(s) for s in specs.values()):
        raise ValueError("V4.1 mixed draft/foreign cache resources are not supported yet")
    result = []
    for key in CacheGroup:
        members = {name: spec for name, spec in specs.items() if group_key(spec) == key}
        if not members:
            continue
        uniform = UniformTypeKVCacheSpecs.from_specs(members)
        if uniform is None:
            raise ValueError(f"Incompatible V4.1 resource layouts in {key.value}")
        result.append(uniform)
    return result


def make_cache_groups(grouped_specs):
    return [KVCacheGroupSpec(layer_names=list(s.kv_cache_specs), kv_cache_spec=s) for s in grouped_specs]


def has_v41_groups(groups):
    return any(
        is_v41_spec(s)
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        for s in g.kv_cache_spec.kv_cache_specs.values()
    )


def pool_bytes_per_block(groups):
    return sum(g.kv_cache_spec.page_size_bytes for g in groups)


def request_blocks(vllm_config, groups):
    # Different logical groups consume different IDs in one global block pool.
    return sum(
        max(
            (s.max_memory_usage_bytes(vllm_config) + s.page_size_bytes - 1) // s.page_size_bytes
            for s in g.kv_cache_spec.kv_cache_specs.values()
        )
        for g in groups
    )


def allocate_cache_config(vllm_config, groups, available_memory):
    """One independent backing tensor per owner; no cross-group pool aliasing.

    UniformType groups can contain different-sized resource planes. All groups
    share the global block ID space, but each plane accounts for its own bytes.
    Thus no common-page padding is necessary for this deliberately conservative
    non-packed allocator. Physical block b has a separate location in each plane.
    """
    specs = {name: spec for g in groups for name, spec in g.kv_cache_spec.kv_cache_specs.items()}
    bytes_per_block = sum(s.page_size_bytes for s in specs.values())
    capacity = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, capacity)
    if num_blocks <= 1 or num_blocks > capacity:
        raise ValueError("Insufficient V4.1 cache memory (including reserved null block), or unsafe block override")
    return num_blocks, [
        KVCacheTensor(size=s.page_size_bytes * num_blocks, shared_by=[name]) for name, s in specs.items()
    ]


def reshape_cache(raw: torch.Tensor, spec):
    if raw.numel() % spec.page_size_bytes:
        raise ValueError("V4.1 cache allocation is not a whole number of pages")
    num_blocks = raw.numel() // spec.page_size_bytes
    # Explicit page stride also handles allocator padding without flatten-copy.
    elements_per_page = spec.page_size_bytes // torch.empty((), dtype=spec.dtype).element_size()
    view = raw.view(spec.dtype).view(num_blocks, elements_per_page)
    elements = spec.storage_block_size * spec.num_kv_heads * spec.head_size
    return view[:, :elements].view(num_blocks, spec.storage_block_size, spec.num_kv_heads, spec.head_size)


def validate_cache_runtime(vllm_config):
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError("V4.1 cache initialization currently requires model runner V1")
    if not vllm_config.model_config.enforce_eager:
        raise NotImplementedError("V4.1 cache initialization currently requires enforce_eager")
    if vllm_config.cache_config.enable_prefix_caching:
        raise NotImplementedError("V4.1 prefix state restoration is not implemented")
    if vllm_config.speculative_config is not None or vllm_config.kv_transfer_config is not None:
        raise NotImplementedError("V4.1 speculative decoding and KV transfer are not implemented")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
            "tensor_parallel_size",
        )
    ):
        raise NotImplementedError("V4.1 initial parameter/cache graph requires TP=PP=DCP=PCP=1")
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("V4.1 requires the hybrid KV cache manager")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("V4.1 initial cache layout requires BF16")
