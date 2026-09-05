# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from vllm_ascend.models.deepseek_v41.layer_plan import build_layer_plan
from vllm_ascend.models.deepseek_v41.model import DeepseekV41ModelContext


@pytest.fixture
def text_config() -> dict:
    return {
        "num_hidden_layers": 40,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "kv_source_layers": [2, 8, 14, 20],
        "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "index_topk": 512,
        "engram_layer_ids": [1, 14],
    }


def test_builds_expected_source_groups(text_config: dict):
    topology = build_layer_plan(text_config)

    assert topology.kv_consumers(2) == tuple(range(2, 8))
    assert topology.kv_consumers(8) == tuple(range(8, 14))
    assert topology.kv_consumers(14) == tuple(range(14, 20))
    assert topology.kv_consumers(20) == tuple(range(20, 40))

    assert topology.index_consumers(20) == tuple(range(20, 24))
    assert topology.index_consumers(24) == tuple(range(24, 28))
    assert topology.index_consumers(36) == tuple(range(36, 40))


def test_layer_26_resolves_layer_20_kv_and_layer_24_index(text_config: dict):
    context = DeepseekV41ModelContext(text_config)
    topology = context.topology
    modules = [object() for _ in topology.layers]
    for layer_idx, module in enumerate(modules):
        context.bind_attention(layer_idx, module)

    sources = context.sources_for(26)
    assert sources.kv_source is modules[20]
    assert sources.index_source is modules[24]
    assert sources.candidate_source is modules[20]
    assert sources.role.compress_ratio == 1
    assert sources.role.uses_candidate_filter


def test_source_roles_and_engram_slots(text_config: dict):
    topology = build_layer_plan(text_config)

    assert topology.layer(2).is_kv_source
    assert topology.layer(2).is_index_source
    assert topology.layer(20).is_candidate_source
    assert topology.layer(1).engram_slot == 0
    assert topology.layer(14).engram_slot == 1
    assert topology.layer(0).kv_source_layer is None


def test_rejects_ratio_mismatch(text_config: dict):
    broken = dict(text_config)
    broken["compress_ratios"] = list(text_config["compress_ratios"])
    broken["compress_ratios"][8] = 1
    with pytest.raises(ValueError, match="KV source"):
        build_layer_plan(broken)
