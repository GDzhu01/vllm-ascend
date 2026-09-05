# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent ownership and page-layout contract for the V4.1 backbone."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .layer_plan import DeepseekV41Topology, _read, build_layer_plan


class CacheKind(str, Enum):
    SWA = "swa_cache"
    LONG = "long_kv_cache"
    INDEX = "index_k_cache"
    TAIL = "compressor_state"


class CacheGroup(str, Enum):
    SWA = "swa"
    COMPRESSED = "full_ratio2"
    FULL = "full_ratio1"
    TAIL = "tail"


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
        owner = layer if kind in (CacheKind.SWA, CacheKind.TAIL) else role.kv_source_layer
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
            add(role.layer_idx, CacheKind.TAIL, CacheGroup.TAIL, ratio, 2 * head_size, ratio, "float32")
    return DeepseekV41CachePlan(topology, tuple(resources), prefix)
