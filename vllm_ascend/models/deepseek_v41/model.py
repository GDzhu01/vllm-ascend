# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction context for the DeepSeek V4.1 Ascend model.

The executable ``nn.Module`` model will be added on top of this context.  The
context is deliberately device-free so model inspection, KV-cache planning and
unit tests all consume exactly the same layer topology before any weights or
device buffers are allocated.
"""

from __future__ import annotations

from typing import Any

from .attention import DeepseekV41AttentionGraph, DeepseekV41AttentionSources
from .layer_plan import DeepseekV41LayerRole, DeepseekV41Topology, build_layer_plan


class DeepseekV41ModelContext:
    """Model-wide topology shared by decoder construction and cache planning."""

    def __init__(self, text_config: Any) -> None:
        self.topology: DeepseekV41Topology = build_layer_plan(text_config)
        self.attention_graph = DeepseekV41AttentionGraph(self.topology)

    def role(self, layer_idx: int) -> DeepseekV41LayerRole:
        return self.topology.layer(layer_idx)

    def bind_attention(self, layer_idx: int, attention: Any) -> None:
        """Publish a constructed attention module when it owns shared state."""
        self.attention_graph.publish_layer(layer_idx, attention)

    def sources_for(self, layer_idx: int) -> DeepseekV41AttentionSources:
        """Resolve the modules a decoder layer must read during forward."""
        return self.attention_graph.resolve_layer(layer_idx)

    def create_caches(self, vllm_config: Any, prefix: str = "model"):
        """Construct once and attach the returned nn.Module to the owning model."""
        from .cache_layer import DeepseekV41ModelCaches

        caches = DeepseekV41ModelCaches(vllm_config, prefix)
        if caches.plan.topology != self.topology:
            # Validate before exposing resources from a mismatched config.
            for module in caches.owners:
                del vllm_config.compilation_config.static_forward_context[module.prefix]
            raise ValueError("Model and KV cache topology differ")
        return caches
