# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level contracts for DeepSeek V4.1 shared sparse attention.

The device implementation will consume this topology instead of inferring
ownership from a per-layer compression ratio, which is the key architectural
difference from DeepSeek V4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .layer_plan import DeepseekV41LayerRole, DeepseekV41Topology


@dataclass(frozen=True)
class DeepseekV41AttentionSources:
    """References resolved for one consumer attention layer."""

    role: DeepseekV41LayerRole
    kv_source: Any | None
    index_source: Any | None
    candidate_source: Any | None


class DeepseekV41AttentionGraph:
    """Resolves source modules without storing request-specific state."""

    def __init__(self, topology: DeepseekV41Topology) -> None:
        self.topology = topology
        self._kv_sources: dict[int, Any] = {}
        self._index_sources: dict[int, Any] = {}
        self._candidate_source: Any | None = None

    def publish_layer(self, layer_idx: int, attention: Any) -> None:
        role = self.topology.layer(layer_idx)
        if role.is_kv_source:
            self._kv_sources[layer_idx] = attention
        if role.is_index_source:
            self._index_sources[layer_idx] = attention
        if role.is_candidate_source:
            self._candidate_source = attention

    def resolve_layer(self, layer_idx: int) -> DeepseekV41AttentionSources:
        role = self.topology.layer(layer_idx)
        kv_source = None if role.kv_source_layer is None else self._kv_sources.get(role.kv_source_layer)
        index_source = None if role.index_source_layer is None else self._index_sources.get(role.index_source_layer)
        if role.has_long_context and kv_source is None:
            raise RuntimeError(f"KV source layer {role.kv_source_layer} was not published before layer {layer_idx}")
        if role.has_long_context and index_source is None:
            raise RuntimeError(
                f"Index source layer {role.index_source_layer} was not published before layer {layer_idx}"
            )
        return DeepseekV41AttentionSources(
            role=role,
            kv_source=kv_source,
            index_source=index_source,
            candidate_source=self._candidate_source if role.uses_candidate_filter else None,
        )
