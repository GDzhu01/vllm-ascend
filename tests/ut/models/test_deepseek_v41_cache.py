# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import compressed_slot_mapping, tail_slot_mapping
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor
from vllm_ascend.models.deepseek_v41.kv_cache import (
    CacheGroup,
    CacheKind,
    DeepseekV41ModelCaches,
    allocate_cache_config,
    build_cache_plan,
    build_cache_specs,
    group_cache_specs,
    make_cache_groups,
    reshape_cache,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Attention


@pytest.fixture
def config():
    # Deliberately small parameter dimensions; source topology matches the backbone.
    return dict(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layers=[2, 8, 14, 20],
        index_source_layers=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer=20,
        candidate_topk_blocks=16,
        candidate_block_size=8,
        index_topk=8,
        engram_layer_ids=[1, 14],
        sliding_window=128,
        head_dim=8,
        index_head_dim=4,
        hidden_size=16,
        num_attention_heads=4,
        index_n_heads=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        rms_norm_eps=1e-6,
    )


@pytest.fixture
def runtime(config):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, enforce_eager=True),
        cache_config=SimpleNamespace(
            block_size=64, enable_prefix_caching=False, cache_dtype="auto", num_gpu_blocks_override=None
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=False,
    )


def test_owner_counts_nested_config_and_source_resolution(config):
    plan = build_cache_plan({"text_config": config}, 64)
    assert len(plan.resources) == 51
    assert Counter(r.kind for r in plan.resources) == {
        CacheKind.SWA: 40,
        CacheKind.LONG: 4,
        CacheKind.INDEX: 4,
        CacheKind.TAIL: 3,
    }
    assert [len(members) for members in plan.groups().values()] == [40, 6, 2, 3]
    assert plan.resource(26, CacheKind.INDEX).owner == 20
    assert plan.resource(26, CacheKind.LONG).owner == 20
    assert plan.resource(26, CacheKind.SWA).owner == 26
    assert plan.resource(20, CacheKind.LONG).storage_rows == 64
    assert plan.resource(2, CacheKind.LONG).storage_rows == 32
    with pytest.raises(ValueError, match="no compressor_state"):
        plan.resource(20, CacheKind.TAIL)


@pytest.mark.parametrize("block_size", [0, -2, 3, 63])
def test_invalid_block_sizes(config, block_size):
    with pytest.raises(ValueError, match="multiple of two"):
        build_cache_plan(config, block_size)


def test_four_groups_and_exact_physical_accounting(config, runtime):
    specs = build_cache_specs(build_cache_plan(config, 64))
    uniform = group_cache_specs(specs)
    assert [len(g.kv_cache_specs) for g in uniform] == [40, 6, 2, 3]
    assert [g.block_size for g in uniform] == [64] * 4
    bytes_per_block = sum(s.page_size_bytes for s in specs.values())
    blocks, tensors = allocate_cache_config(runtime, make_cache_groups(uniform), bytes_per_block * 10 + 1)
    assert blocks == 10
    assert len(tensors) == 51
    assert sum(t.size for t in tensors) == 10 * bytes_per_block
    assert len({t.shared_by[0] for t in tensors}) == 51
    assert all(len(t.shared_by) == 1 for t in tensors)
    for allocation in tensors:
        spec = specs[allocation.shared_by[0]]
        raw = torch.zeros(allocation.size, dtype=torch.uint8)
        cache = reshape_cache(raw, spec)
        assert cache.shape == (blocks, spec.storage_block_size, 1, spec.head_size)
        assert cache.data_ptr() == raw.data_ptr()


def test_mixed_layouts_rejected(config):
    specs = build_cache_specs(build_cache_plan(config, 64))
    specs["foreign"] = object()
    with pytest.raises(ValueError, match="foreign"):
        group_cache_specs(specs)


def test_unsafe_override_rejected(config, runtime):
    groups = make_cache_groups(group_cache_specs(build_cache_specs(build_cache_plan(config, 64))))
    runtime.cache_config.num_gpu_blocks_override = 100
    with pytest.raises(ValueError, match="unsafe block override"):
        allocate_cache_config(runtime, groups, 1)


def test_model_registration_and_binding(runtime):
    caches = DeepseekV41ModelCaches(runtime, "language_model.model")
    assert len(runtime.compilation_config.static_forward_context) == 51
    assert len(list(caches.named_buffers())) == 0  # No max-sequence allocation in __init__.
    assert caches.resolve(26, CacheKind.INDEX) is caches.resolve(20, CacheKind.INDEX)
    assert caches.resolve(26, CacheKind.SWA) is not caches.resolve(20, CacheKind.SWA)
    assert all(module.kv_cache[0].numel() == 0 for module in caches.owners)
    with pytest.raises(ValueError, match="Duplicate"):
        DeepseekV41ModelCaches(runtime, "language_model.model")
    assert len(runtime.compilation_config.static_forward_context) == 51
    tail = caches.resolve(2, CacheKind.TAIL)
    tail.kv_cache = [torch.ones((4, 2, 1, 16), dtype=torch.float32)]
    caches.reset_tail(2, torch.tensor([2]))
    assert (tail.kv_cache[0][2, :, :, :8] == 0).all()
    assert torch.isneginf(tail.kv_cache[0][2, :, :, 8:]).all()
    assert (tail.kv_cache[0][1] == 1).all()


@pytest.mark.parametrize("feature", ["prefix", "spec", "pd", "pp", "v2", "graph"])
def test_unsupported_runtime_fails_before_registration(runtime, feature):
    if feature == "prefix":
        runtime.cache_config.enable_prefix_caching = True
    elif feature == "spec":
        runtime.speculative_config = object()
    elif feature == "pd":
        runtime.kv_transfer_config = object()
    elif feature == "pp":
        runtime.parallel_config.pipeline_parallel_size = 2
    elif feature == "v2":
        runtime.use_v2_model_runner = True
    else:
        runtime.model_config.enforce_eager = False
    with pytest.raises(NotImplementedError):
        DeepseekV41ModelCaches(runtime)
    assert not runtime.compilation_config.static_forward_context


def test_compression_slot_mapping():
    slots = torch.tensor([-1, 0, 1, 62, 63, 320, 321, 383])
    assert compressed_slot_mapping(slots, 2).tolist() == [-1, -1, 0, -1, 31, -1, 160, 191]
    assert torch.equal(compressed_slot_mapping(slots, 1), slots)


def test_tail_slots_cross_logical_blocks_and_request_reordering():
    # Request 0: absolute positions 128,129; request 1: position 1025.
    starts = torch.tensor([0, 2, 3], dtype=torch.int32)
    lengths = torch.tensor([130, 1026], dtype=torch.int32)
    table = torch.tensor([[7], [3]], dtype=torch.int32)
    assert tail_slot_mapping(table, starts, lengths, 5).tolist() == [14, 15, 7, -1, -1]


def test_actual_attention_parameter_ownership(config):
    plan = build_cache_plan(config, 64)
    layers = {i: DeepseekV41Attention(config, i, plan) for i in (0, 2, 20, 24, 26)}
    assert layers[0].compressor is None and layers[0].indexer is None
    assert hasattr(layers[2].compressor, "wgate")
    assert not hasattr(layers[20].compressor, "wgate")
    assert layers[20].indexer.owns_k
    assert not layers[24].indexer.owns_k
    assert not hasattr(layers[24].indexer, "wk")
    assert layers[26].indexer is None
    assert layers[26].cache_prefixes[CacheKind.INDEX] == layers[20].cache_prefixes[CacheKind.INDEX]
    assert layers[2].project_output(torch.zeros(3, 4, 8, dtype=torch.bfloat16)).shape == (3, 16)
    with pytest.raises(NotImplementedError, match="not connected"):
        layers[2](torch.zeros(1, 16))


@pytest.mark.parametrize("chunks", [(1, 1, 1, 2, 2), (3, 4), (2, 2, 3), (7,)])
@torch.inference_mode()
def test_compressor_chunk_boundary_matches_vector_reference(config, chunks):
    torch.manual_seed(7)
    compressor = DeepseekV41Compressor(config, 2)
    x = torch.randn(7, 16, dtype=torch.bfloat16)
    kv = compressor.wkv(x.float())[:6].reshape(3, 2, 8)
    gate = compressor.wgate(x.float())[:6].reshape(3, 2, 8)
    expected = compressor.norm((kv * gate.softmax(dim=1)).sum(dim=1).to(x.dtype))
    state = torch.zeros(2, 16, dtype=torch.float32)
    state[:, 8:] = -torch.inf
    actual = []
    start = 0
    for size in chunks:
        actual.append(compressor(x[start : start + size], start, state))
        start += size
    torch.testing.assert_close(torch.cat(actual), expected)
    torch.testing.assert_close(state[0, :8], compressor.wkv(x[-1:].float())[0])


def test_tail_is_fixed_request_private(config):
    plan = build_cache_plan(config, 64)
    specs = build_cache_specs(plan)
    tail = specs[plan.resource(2, CacheKind.TAIL).name]
    assert not tail.participates_in_prefix_caching
    assert tail.max_num_blocks_per_req(None, 1048576) == 1
    assert tail.max_memory_usage_bytes(None) == tail.page_size_bytes
    assert tail.storage_block_size == 2
    assert CacheGroup.TAIL in plan.groups()
