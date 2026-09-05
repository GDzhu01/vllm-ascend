# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-side owner registration and source resolution, without global state."""

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from .cache_plan import CacheKind, build_cache_plan
from .kv_cache import build_cache_specs


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
        self.owners = nn.ModuleList([DeepseekV41CacheLayer(name, spec) for name, spec in specs.items()])
        self._owner_indices = {module.prefix: idx for idx, module in enumerate(self.owners)}
        context.update({module.prefix: module for module in self.owners})

    def resolve(self, layer: int, kind: CacheKind) -> DeepseekV41CacheLayer:
        name = self.plan.resource(layer, kind).name
        return self.owners[self._owner_indices[name]]

    def reset_tail(self, layer: int, block_ids: torch.Tensor):
        """Explicit lifecycle hook: call on admission/recompute, never each step."""
        cache = self.resolve(layer, CacheKind.TAIL).kv_cache[0]
        width = cache.shape[-1] // 2
        cache[block_ids, :, :, :width] = 0
        cache[block_ids, :, :, width:] = -torch.inf
