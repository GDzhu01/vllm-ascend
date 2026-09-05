# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.dsa_v41 import DeepseekV41MetadataBuilder, compressed_slot_mapping
from vllm_ascend.core.deepseek_v41 import (
    allocate_cache_config,
    group_cache_specs,
    make_cache_groups,
    reshape_cache,
)
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor
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


def construct_layers(runtime, prefix="model"):
    config = runtime.model_config.hf_text_config
    return [
        DeepseekV41Attention(config, i, runtime, f"{prefix}.layers.{i}.self_attn")
        for i in range(config["num_hidden_layers"])
    ]


def collect_specs(runtime):
    construct_layers(runtime)
    return {
        name: module.get_kv_cache_spec(runtime)
        for name, module in runtime.compilation_config.static_forward_context.items()
    }


def test_owner_counts_nested_config_and_source_resolution(config, runtime):
    layers = construct_layers(runtime)
    context = runtime.compilation_config.static_forward_context
    assert len(context) == 51
    assert sum(hasattr(layer, "long_kv_cache") for layer in layers) == 4
    assert sum(layer.indexer is not None and hasattr(layer.indexer, "k_cache") for layer in layers) == 4
    assert sum(layer.compressor is not None and hasattr(layer.compressor, "state_cache") for layer in layers) == 3
    assert context[layers[26].long_kv_source_prefix] is layers[20].long_kv_cache
    assert context[layers[26].index_k_source_prefix] is layers[20].indexer.k_cache
    assert layers[26].swa_cache is not layers[20].swa_cache
    assert layers[20].long_kv_cache.spec.storage_block_size == 64
    assert layers[2].long_kv_cache.spec.storage_block_size == 32
    assert not hasattr(layers[20].compressor, "state_cache")
    with pytest.raises(ValueError, match="Duplicate"):
        DeepseekV41Attention({"text_config": config}, 2, runtime)
    assert len(context) == 51


@pytest.mark.parametrize("block_size", [0, -2, 3, 63])
def test_invalid_block_sizes(config, runtime, block_size):
    runtime.cache_config.block_size = block_size
    with pytest.raises(ValueError, match="multiple of two"):
        DeepseekV41Attention(config, 0, runtime)


def test_four_groups_and_exact_physical_accounting(config, runtime):
    specs = collect_specs(runtime)
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


def test_mixed_layouts_rejected(config, runtime):
    specs = collect_specs(runtime)
    specs["foreign"] = object()
    with pytest.raises(ValueError, match="foreign"):
        group_cache_specs(specs)


def test_unsafe_override_rejected(config, runtime):
    groups = make_cache_groups(group_cache_specs(collect_specs(runtime)))
    runtime.cache_config.num_gpu_blocks_override = 100
    with pytest.raises(ValueError, match="unsafe block override"):
        allocate_cache_config(runtime, groups, 1)


def test_model_registration_and_binding(runtime):
    layers = construct_layers(runtime, "language_model.model")
    context = runtime.compilation_config.static_forward_context
    assert len(context) == 51
    assert all(module.kv_cache[0].numel() == 0 for module in context.values())
    state = layers[2].compressor.state_cache
    assert state is context["language_model.model.layers.2.self_attn.compressor.state_cache"]
    assert state.spec.sliding_window == 2
    assert state.spec.storage_block_size == 64
    assert state.state_dim == 16
    # Each cache is registered once and belongs to its actual component.
    modules = torch.nn.ModuleList(layers)
    owned_names = [name for name, module in modules.named_modules() if hasattr(module, "kv_cache")]
    assert len(owned_names) == 51
    assert "2.compressor.state_cache" in owned_names
    assert "2.indexer.k_cache" in owned_names
    assert "2.long_kv_cache" in owned_names


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
        DeepseekV41Attention(runtime.model_config.hf_text_config, 0, runtime)
    assert not runtime.compilation_config.static_forward_context


def test_compression_slot_mapping():
    slots = torch.tensor([-1, 0, 1, 62, 63, 320, 321, 383])
    assert compressed_slot_mapping(slots, 2).tolist() == [-1, -1, 0, -1, 31, -1, 160, 191]
    assert torch.equal(compressed_slot_mapping(slots, 1), slots)


def test_state_metadata_keeps_original_token_slots(config, runtime):
    specs = collect_specs(runtime)
    spec = specs["model.layers.2.self_attn.compressor.state_cache"]
    builder = DeepseekV41MetadataBuilder(spec, [], runtime, torch.device("cpu"))
    slots = torch.tensor([7 * 64 + 63, 3 * 64, -1])
    common = SimpleNamespace(
        slot_mapping=slots,
        block_table_tensor=torch.tensor([[7, 3]]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([65]),
    )
    metadata = builder.build(0, common)
    assert metadata.is_compressor_state
    assert metadata.slot_mapping is slots
    assert metadata.compress_ratio == 1
    assert metadata.storage_block_size == 64


def test_actual_attention_parameter_ownership(config, runtime):
    layers = {i: DeepseekV41Attention(config, i, runtime) for i in (0, 2, 20, 24, 26)}
    assert layers[0].compressor is None and layers[0].indexer is None
    assert hasattr(layers[2].compressor, "wgate")
    assert not hasattr(layers[20].compressor, "wgate")
    assert layers[20].indexer.owns_k
    assert not layers[24].indexer.owns_k
    assert not hasattr(layers[24].indexer, "wk")
    assert layers[26].indexer is None
    assert layers[26].index_k_source_prefix == layers[20].index_k_source_prefix
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
    # Each successive group occupies another page. Noncontiguous physical IDs
    # and chunks spanning those pages catch the old circular-buffer addressing.
    state = torch.full((6, 2, 16), float("nan"), dtype=torch.float32)
    block_table = [4, 1, 5, 2]
    actual = []
    start = 0
    for size in chunks:
        actual.append(compressor(x[start : start + size], start, state, block_table))
        start += size
    torch.testing.assert_close(torch.cat(actual), expected)
    torch.testing.assert_close(state[2, 0, :8], compressor.wkv(x[-1:].float())[0])


def test_state_uses_swa_memory_and_block_table_rules(config, runtime):
    from vllm_ascend.core.deepseek_v41 import DeepseekV41CompressorStateSpec
    from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec

    spec = collect_specs(runtime)["model.layers.2.self_attn.compressor.state_cache"]
    assert isinstance(spec, AscendSlidingWindowMLASpec)
    assert isinstance(spec, DeepseekV41CompressorStateSpec)
    assert spec.sliding_window == 2
    assert spec.compress_ratio == 1
    assert spec.storage_block_size == 64
    assert spec.page_size_bytes == 64 * 16 * 4
    assert spec.max_num_blocks_per_req(runtime, 1024) == 16
    runtime.max_in_flight_tokens = 128
    runtime.model_config.max_model_len = 1024
    expected_pages = spec.max_admission_blocks_per_request(128, 1024)
    assert spec.max_memory_usage_bytes(runtime) == expected_pages * spec.page_size_bytes
    assert expected_pages > 1
    assert spec.sliding_window != config["sliding_window"]


@torch.inference_mode()
def test_compressor_rejects_missing_previous_state_page(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 2, 16), float("nan"), dtype=torch.float32)
    with pytest.raises(ValueError, match="absent/null"):
        compressor(torch.zeros(1, 16, dtype=torch.bfloat16), 1, state, [0])


@torch.inference_mode()
def test_state_page_reuse_does_not_require_request_reset(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 2, 16), float("nan"), dtype=torch.float32)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    expected = compressor(x, 0, state, [1]).clone()
    state[1].fill_(12345)
    actual = compressor(x, 0, state, [1])
    torch.testing.assert_close(actual, expected)


def test_state_registers_standard_sliding_window_manager(monkeypatch):
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    from vllm_ascend.core.deepseek_v41 import DeepseekV41CompressorStateSpec
    from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs

    registrations = {}

    def record(kvcache_spec_cls, manager_class, uniform_type_base_spec):
        registrations[kvcache_spec_cls] = manager_class

    monkeypatch.setattr(KVCacheSpecRegistry, "register", record)
    register_ascend_kv_cache_specs()
    assert registrations[DeepseekV41CompressorStateSpec] is SlidingWindowManager
    manager = SimpleNamespace(sliding_window=2)
    for computed in (1, 63, 64, 65, 128, 129):
        # At the next query, the immediately previous token is never skipped.
        assert SlidingWindowManager.get_num_skipped_tokens(manager, computed) == computed - 1


@torch.inference_mode()
def test_interleaved_request_state_isolation(config):
    compressor = DeepseekV41Compressor(config, 2)
    state = torch.full((3, 2, 16), float("nan"), dtype=torch.float32)
    first = torch.randn(2, 16, dtype=torch.bfloat16)
    second = torch.randn(2, 16, dtype=torch.bfloat16)
    compressor(first[:1], 0, state, [1])
    saved = state[1, 0].clone()
    compressor(second, 0, state, [2])
    torch.testing.assert_close(state[1, 0], saved)
    actual = compressor(first[1:], 1, state, [1])
    expected = compressor(first, 0, state, [1])
    torch.testing.assert_close(actual, expected)
