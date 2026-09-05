# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 attention parameter graph and source/cache references.

Full backbone/LM execution and checkpoint loading are not implemented yet.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .compressor import DeepseekV41Compressor, DeepseekV41RMSNorm
from .indexer import DeepseekV41Indexer
from .kv_cache import CacheKind, _read, text_config_of


class DeepseekV41Attention(nn.Module):
    """Parameter ownership and cache references for a single attention layer.

    Construct caches once on the parent model; retain only resource prefix
    strings here, never a second registration of the shared cache module.
    """

    def __init__(self, config, layer_idx, cache_plan):
        super().__init__()
        config = text_config_of(config)
        role = cache_plan.topology.layer(layer_idx)
        self.role = role
        dim = _read(config, "hidden_size")
        heads = _read(config, "num_attention_heads")
        width = _read(config, "head_dim")
        q_rank = _read(config, "q_lora_rank")
        o_rank = _read(config, "o_lora_rank")
        groups = _read(config, "o_groups")
        if heads % groups:
            raise ValueError("Attention heads must be divisible by output groups")
        self.n_groups = groups
        self.attn_sink = nn.Parameter(torch.empty(heads, dtype=torch.float32))
        self.wq_a = nn.Linear(dim, q_rank, bias=False, dtype=torch.bfloat16)
        self.q_norm = DeepseekV41RMSNorm(q_rank, _read(config, "rms_norm_eps"))
        self.wq_b = nn.Linear(q_rank, heads * width, bias=False, dtype=torch.bfloat16)
        self.wkv = nn.Linear(dim, width, bias=False, dtype=torch.bfloat16)
        self.kv_norm = DeepseekV41RMSNorm(width, _read(config, "rms_norm_eps"))
        # Same flattened checkpoint layout as V4; apply separately to each group.
        self.wo_a = nn.Linear(heads * width // groups, groups * o_rank, bias=False, dtype=torch.bfloat16)
        self.wo_b = nn.Linear(groups * o_rank, dim, bias=False, dtype=torch.bfloat16)
        self.compressor = DeepseekV41Compressor(config, role.compress_ratio) if role.is_kv_source else None
        self.indexer = DeepseekV41Indexer(config, role.is_kv_source) if role.is_index_source else None
        kinds = [CacheKind.SWA]
        if role.has_long_context:
            kinds.extend((CacheKind.LONG, CacheKind.INDEX))
        if role.is_kv_source and role.compress_ratio == 2:
            kinds.append(CacheKind.TAIL)
        self.cache_prefixes = {kind: cache_plan.resource(layer_idx, kind).name for kind in kinds}

    def project_output(self, attention_output):
        x = attention_output.reshape(-1, self.n_groups, self.wo_a.in_features)
        weight = self.wo_a.weight.reshape(self.n_groups, -1, self.wo_a.in_features)
        projected = torch.einsum("tgi,goi->tgo", x, weight)
        return F.linear(projected.flatten(1), self.wo_b.weight)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "V4.1 parameter/cache graph is available; paged sparse attention is not connected yet"
        )
