# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 cache ownership, registration, hybrid grouping and allocation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


@dataclass(frozen=True)
class DeepseekV41LayerRole:
    """The attention and Engram responsibilities of one backbone layer."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool
    engram_slot: int | None

    @property
    def has_long_context(self) -> bool:
        return self.compress_ratio > 0


@dataclass(frozen=True)
class DeepseekV41Topology:
    """Validated, immutable model-wide source/consumer topology."""

    layers: tuple[DeepseekV41LayerRole, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(role.layer_idx for role in self.layers if role.index_source_layer == source_layer)


def _read(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        try:
            return config[name]
        except KeyError as exc:
            raise ValueError(f"DeepSeek V4.1 config is missing {name!r}") from exc
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"DeepSeek V4.1 config is missing {name!r}") from exc


def _as_int_tuple(config: Any, name: str) -> tuple[int, ...]:
    value = _read(config, name)
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"DeepSeek V4.1 {name} must be a list of integers")
    return tuple(value)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    return next((source for source in reversed(sources) if source <= layer_idx), None)


def build_layer_plan(config: Any) -> DeepseekV41Topology:
    """Build and validate the V4.1 layer-sharing graph from a text config.

    ``config`` may be a Transformers config object or the raw ``text_config``
    dictionary.  Extra compression ratios for speculative layers are allowed,
    but only the first ``num_hidden_layers`` entries describe the backbone.
    """

    num_layers = int(_read(config, "num_hidden_layers"))
    ratios = _as_int_tuple(config, "compress_ratios")
    kv_sources = _as_int_tuple(config, "kv_source_layers")
    index_sources = _as_int_tuple(config, "index_source_layers")
    engram_layers = _as_int_tuple(config, "engram_layer_ids")
    candidate_source = int(_read(config, "candidate_source_layer"))
    candidate_topk_blocks = int(_read(config, "candidate_topk_blocks"))
    candidate_block_size = int(_read(config, "candidate_block_size"))
    index_topk = int(_read(config, "index_topk"))

    if num_layers <= 0:
        raise ValueError("DeepSeek V4.1 num_hidden_layers must be positive")
    if len(ratios) < num_layers:
        raise ValueError(
            "DeepSeek V4.1 compress_ratios must cover every backbone layer: "
            f"got {len(ratios)} ratios for {num_layers} layers"
        )
    ratios = ratios[:num_layers]
    if any(ratio not in (0, 1, 2) for ratio in ratios):
        raise ValueError(f"DeepSeek V4.1 backbone only supports compression ratios 0, 1 and 2; got {ratios}")

    for name, sources in (("kv_source_layers", kv_sources), ("index_source_layers", index_sources)):
        if tuple(sorted(set(sources))) != sources:
            raise ValueError(f"DeepSeek V4.1 {name} must be sorted and unique")
        if any(source < 0 or source >= num_layers for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} contains a layer outside the backbone")
        if any(ratios[source] == 0 for source in sources):
            raise ValueError(f"DeepSeek V4.1 {name} cannot point to a local-only layer")

    if not set(kv_sources).issubset(index_sources):
        raise ValueError("Every DeepSeek V4.1 KV source must also be an index source")
    if candidate_source not in kv_sources:
        raise ValueError("DeepSeek V4.1 candidate_source_layer must be a KV source")
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0 or index_topk <= 0:
        raise ValueError("DeepSeek V4.1 candidate and index TopK values must be positive")
    if len(set(engram_layers)) != len(engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids must be unique")
    if any(layer < 0 or layer >= num_layers for layer in engram_layers):
        raise ValueError("DeepSeek V4.1 engram_layer_ids contains a layer outside the backbone")

    engram_slots = {layer_idx: slot for slot, layer_idx in enumerate(engram_layers)}
    roles: list[DeepseekV41LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        kv_source = _latest_source(layer_idx, kv_sources) if ratio else None
        index_source = _latest_source(layer_idx, index_sources) if ratio else None
        if ratio and (kv_source is None or index_source is None):
            raise ValueError(f"DeepSeek V4.1 layer {layer_idx} has long-context attention but no source layer")
        if kv_source is not None and ratios[kv_source] != ratio:
            raise ValueError(
                f"DeepSeek V4.1 layer {layer_idx} has ratio {ratio}, but its KV source "
                f"layer {kv_source} has ratio {ratios[kv_source]}"
            )

        roles.append(
            DeepseekV41LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # Consumer layers inherit the selection policy of their index
                # source.  For example, layer 26 reuses layer 24 TopK, and that
                # TopK was computed inside layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
                engram_slot=engram_slots.get(layer_idx),
            )
        )

    return DeepseekV41Topology(
        layers=tuple(roles),
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        index_topk=index_topk,
    )


class CacheKind(str, Enum):
    SWA = "swa_cache"
    LONG = "long_kv_cache"
    INDEX = "index_k_cache"
    STATE = "compressor.state_cache"


class CacheGroup(str, Enum):
    SWA = "swa"
    COMPRESSED = "full_ratio2"
    FULL = "full_ratio1"
    STATE = "compressor_state"


def text_config_of(config: Any) -> Any:
    if isinstance(config, dict):
        return config.get("text_config", config)
    return getattr(config, "text_config", config)


@dataclass(frozen=True)
class DeepseekV41CacheResource:
    name: str
    owner: int
    kind: CacheKind
    group: CacheGroup
    block_size: int  # Original tokens, NEVER compressed rows.
    storage_rows: int
    head_size: int
    dtype: str
    compress_ratio: int
    sliding_window: int = 0

    @property
    def page_size_bytes(self) -> int:
        return self.storage_rows * self.head_size * (4 if self.dtype == "float32" else 2)


@dataclass(frozen=True)
class DeepseekV41CachePlan:
    topology: DeepseekV41Topology
    resources: tuple[DeepseekV41CacheResource, ...]
    prefix: str

    def resource(self, layer: int, kind: CacheKind) -> DeepseekV41CacheResource:
        role = self.topology.layer(layer)
        owner = layer if kind in (CacheKind.SWA, CacheKind.STATE) else role.kv_source_layer
        for resource in self.resources:
            if resource.owner == owner and resource.kind == kind:
                return resource
        raise ValueError(f"Layer {layer} has no {kind.value} resource")

    def groups(self) -> dict[CacheGroup, tuple[DeepseekV41CacheResource, ...]]:
        return {
            group: tuple(r for r in self.resources if r.group == group)
            for group in CacheGroup
            if any(r.group == group for r in self.resources)
        }


def build_cache_plan(config: Any, block_size: int, prefix: str = "model") -> DeepseekV41CachePlan:
    config = text_config_of(config)
    topology = build_layer_plan(config)
    if block_size <= 0 or block_size % 2:
        raise ValueError("V4.1 logical block_size must be a positive multiple of two")
    head_size = int(_read(config, "head_dim"))
    index_size = int(_read(config, "index_head_dim"))
    window = int(_read(config, "sliding_window"))
    if min(head_size, index_size, window) <= 0:
        raise ValueError("V4.1 cache dimensions and sliding window must be positive")
    resources = []

    def add(layer, kind, group, rows, width, ratio=1, dtype="bfloat16", sliding_window=0):
        resources.append(
            DeepseekV41CacheResource(
                name=f"{prefix}.layers.{layer}.self_attn.{kind.value}",
                owner=layer,
                kind=kind,
                group=group,
                block_size=block_size,
                storage_rows=rows,
                head_size=width,
                dtype=dtype,
                compress_ratio=ratio,
                sliding_window=sliding_window,
            )
        )

    for role in topology.layers:
        add(role.layer_idx, CacheKind.SWA, CacheGroup.SWA, block_size, head_size, sliding_window=window)
        if not role.is_kv_source:
            continue
        ratio = role.compress_ratio
        group = CacheGroup.COMPRESSED if ratio == 2 else CacheGroup.FULL
        add(role.layer_idx, CacheKind.LONG, group, block_size // ratio, head_size, ratio)
        add(role.layer_idx, CacheKind.INDEX, group, block_size // ratio, index_size, ratio)
        if ratio == 2:
            # Each row holds concatenated raw KV and gate score; not K and V.
            add(
                role.layer_idx,
                CacheKind.STATE,
                CacheGroup.STATE,
                block_size,
                2 * head_size,
                dtype="float32",
                sliding_window=ratio,
            )
    return DeepseekV41CachePlan(topology, tuple(resources), prefix)


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


def build_cache_specs(plan: DeepseekV41CachePlan):
    specs = {}
    for r in plan.resources:
        kwargs = dict(block_size=r.block_size, num_kv_heads=1, head_size=r.head_size, dtype=getattr(torch, r.dtype))
        if r.kind == CacheKind.SWA:
            spec = DeepseekV41SWASpec(**kwargs, sliding_window=r.sliding_window)
        elif r.kind == CacheKind.STATE:
            spec = DeepseekV41CompressorStateSpec(**kwargs, sliding_window=r.sliding_window)
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


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = [torch.empty(0)]

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        # Lazy import avoids backend / model cache definition cycles.
        from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend

        return DeepseekV41CacheBackend


class DeepseekV41CompressorStateCache(DeepseekV41CacheLayer):
    """State-cache module with the same paged vector layout used by V4.

    Pass kv_cache[0].squeeze(-2) and the state's block table to the compressor.
    The V4 constructor itself cannot be reused: it asserts ratio in (4, 128).
    """

    def __init__(self, prefix, spec):
        if spec.dtype != torch.float32 or spec.compress_ratio != 1 or spec.sliding_window != 2:
            raise ValueError("V4.1 compressor state requires FP32 uncompressed rows and window two")
        super().__init__(prefix, spec)
        self.state_dim = spec.head_size
        self.dtype = spec.dtype
        self.compress_ratio = 2  # Pooling ratio; spec storage ratio remains one.
        self.block_size = spec.block_size
        self.sliding_window = spec.sliding_window


class DeepseekV41ModelCaches(nn.Module):
    """One registered cache module per actual backbone resource, never per reader.

    Attach once to the model. Decoder layers resolve owners by name rather than
    attaching the same nn.Module repeatedly (which would duplicate state paths).
    """

    def __init__(self, vllm_config, prefix="model"):
        super().__init__()
        validate_cache_runtime(vllm_config)
        self.plan = build_cache_plan(
            vllm_config.model_config.hf_text_config, vllm_config.cache_config.block_size, prefix
        )
        specs = build_cache_specs(self.plan)
        context = vllm_config.compilation_config.static_forward_context
        duplicates = specs.keys() & context.keys()
        if duplicates:
            raise ValueError(f"Duplicate V4.1 cache prefixes: {sorted(duplicates)}")
        self.owners = nn.ModuleList(
            [
                (
                    DeepseekV41CompressorStateCache
                    if isinstance(spec, DeepseekV41CompressorStateSpec)
                    else DeepseekV41CacheLayer
                )(name, spec)
                for name, spec in specs.items()
            ]
        )
        self._owner_indices = {module.prefix: idx for idx, module in enumerate(self.owners)}
        context.update({module.prefix: module for module in self.owners})

    def resolve(self, layer: int, kind: CacheKind) -> DeepseekV41CacheLayer:
        name = self.plan.resource(layer, kind).name
        return self.owners[self._owner_indices[name]]
