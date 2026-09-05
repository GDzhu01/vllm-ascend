# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 hybrid specs and conservative, non-aliasing physical planner.

No V4 layout constants or ratio<=1 ownership shortcuts apply here.
"""

from dataclasses import dataclass

import torch
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager, KpoolTailSpec

from .cache_plan import CacheGroup, CacheKind, DeepseekV41CachePlan


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
class DeepseekV41TailSpec(KpoolTailSpec):
    # One request-owned block with two raw slots; block_size stays in raw tokens.
    @property
    def storage_block_size(self):
        return 2

    @property
    def real_page_size_bytes(self):
        return self.storage_block_size * self.num_kv_heads * self.head_size * 4

    def max_memory_usage_bytes(self, vllm_config):
        return self.page_size_bytes

    def is_uniform_with_collection(self, specs):
        return all(isinstance(s, DeepseekV41TailSpec) for s in specs.values())


class DeepseekV41TailManager(KpoolTailManager):
    """Request-private fixed block; no prefix hits or sliding-window eviction.

    The V4.1 model integration rejects PD and speculative decoding. Inheriting
    allocation lifetime does not advertise either execution feature.
    """


def is_v41_spec(spec):
    return isinstance(spec, (DeepseekV41FullSpec, DeepseekV41SWASpec, DeepseekV41TailSpec))


def group_key(spec):
    if isinstance(spec, DeepseekV41SWASpec):
        return CacheGroup.SWA
    if isinstance(spec, DeepseekV41TailSpec):
        return CacheGroup.TAIL
    if isinstance(spec, DeepseekV41FullSpec):
        return CacheGroup.COMPRESSED if spec.compress_ratio == 2 else CacheGroup.FULL
    raise TypeError(f"Not a V4.1 cache spec: {type(spec)}")


def build_cache_specs(plan: DeepseekV41CachePlan):
    specs = {}
    for r in plan.resources:
        kwargs = dict(block_size=r.block_size, num_kv_heads=1, head_size=r.head_size, dtype=getattr(torch, r.dtype))
        if r.kind == CacheKind.SWA:
            spec = DeepseekV41SWASpec(**kwargs, sliding_window=r.sliding_window)
        elif r.kind == CacheKind.TAIL:
            spec = DeepseekV41TailSpec(**kwargs, sliding_window=2)
        else:
            spec = DeepseekV41FullSpec(**kwargs, compress_ratio=r.compress_ratio)
        specs[r.name] = spec
    return specs


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
