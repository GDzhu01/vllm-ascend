# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 model construction components for Ascend.

The V4.1 attention topology is intentionally kept separate from DeepSeek V4.
DeepSeek V4 owns compressor and indexer state per attention layer, while V4.1
publishes those states from a small set of source layers and reuses them from
later consumer layers.
"""

from .layer_plan import DeepseekV41LayerRole, DeepseekV41Topology, build_layer_plan
from .model import DeepseekV41ModelContext

__all__ = [
    "DeepseekV41LayerRole",
    "DeepseekV41Topology",
    "DeepseekV41ModelContext",
    "build_layer_plan",
]
