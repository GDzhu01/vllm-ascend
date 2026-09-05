# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unfused V4.1 attention parameter graph (single-rank bring-up).

These modules construct actual parameters, not source-reference placeholders.
RoPE, reference low-bit rounding, selection, and paged attention execution are
intentionally not replaced by a dense approximation or a zero-output forward.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .cache_plan import CacheKind, _read, text_config_of


class DeepseekV41RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, ratio):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("V4.1 compressor requires ratio 1 or 2")
        self.ratio = ratio
        self.width = _read(config, "head_dim")
        dim = _read(config, "hidden_size")
        self.wkv = nn.Linear(dim, self.width, bias=False, dtype=torch.float32 if ratio == 2 else torch.bfloat16)
        self.norm = DeepseekV41RMSNorm(self.width, _read(config, "rms_norm_eps"))
        if ratio == 2:
            self.wgate = nn.Linear(dim, self.width, bias=False, dtype=torch.float32)

    def forward(self, x, start_pos: int, state=None):
        """Pool one request's chunk; state is its scheduler-owned [2, 2*D] view.

        Reference bring-up path, deliberately sequential. State is not owned by
        batch position. Returns only completed groups, before RoPE.
        """
        if start_pos < 0 or x.ndim != 2:
            raise ValueError("Expected nonnegative start_pos and [tokens, hidden] input")
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        if state is None or state.shape != (2, 2 * self.width) or state.dtype != torch.float32:
            raise ValueError("Ratio2 requires request-owned FP32 [2, 2*head_dim] state")
        kv = self.wkv(x.float())
        score = self.wgate(x.float())
        completed = []
        for token in range(x.shape[0]):
            slot = (start_pos + token) % self.ratio
            state[slot, : self.width] = kv[token]
            state[slot, self.width :] = score[token]
            if slot == self.ratio - 1:
                pooled = (state[:, : self.width] * state[:, self.width :].softmax(dim=0)).sum(dim=0)
                completed.append(pooled)
        latent = torch.stack(completed).to(x.dtype) if completed else x.new_empty((0, self.width))
        return self.norm(latent)


class DeepseekV41Indexer(nn.Module):
    def __init__(self, config, owns_k):
        super().__init__()
        self.owns_k = owns_k
        heads = _read(config, "index_n_heads")
        width = _read(config, "index_head_dim")
        self.wq_b = nn.Linear(_read(config, "q_lora_rank"), heads * width, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(_read(config, "hidden_size"), heads, bias=False, dtype=torch.bfloat16)
        if owns_k:
            self.wk = nn.Linear(_read(config, "head_dim"), width, bias=False, dtype=torch.bfloat16)
            self.k_norm = DeepseekV41RMSNorm(width, _read(config, "rms_norm_eps"))


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
