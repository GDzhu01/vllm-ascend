# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unfused ratio1/ratio2 compressor and shared reference RMS normalization."""

import torch
from torch import nn

from .kv_cache import _read


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

    def forward(self, x, start_pos: int, state_cache=None, state_block_table=None):
        """Pool one request's chunk using V4-style paged KV/score state.

        state_cache: FP32 [pages, state_block_size, 2*D], i.e. the model cache
        with its singleton KV-head axis squeezed, as in the V4 operator call.
        state_block_table: this request's logical-to-physical page IDs, supplied
        as a host list/tuple in this unfused reference path. The fused operator
        will consume the batched device block table directly.
        Returns only completed groups, before RoPE. Every row read was written
        for the request's actual token; recycled pages require no blanket reset.
        """
        if start_pos < 0 or x.ndim != 2:
            raise ValueError("Expected nonnegative start_pos and [tokens, hidden] input")
        if self.ratio == 1:
            return self.norm(self.wkv(x))
        if (
            state_cache is None
            or state_cache.ndim != 3
            or state_cache.shape[-1] != 2 * self.width
            or state_cache.dtype != torch.float32
        ):
            raise ValueError("Ratio2 requires paged FP32 [pages, block_size, 2*head_dim] state")
        if not isinstance(state_block_table, (list, tuple)):
            raise ValueError("Reference compressor requires a host list/tuple state_block_table")
        block_size = state_cache.shape[1]
        if block_size <= 0 or block_size % self.ratio:
            raise ValueError("State block size must be a positive multiple of the pooling ratio")

        def state_row(position):
            logical_block, offset = divmod(position, block_size)
            if logical_block >= len(state_block_table):
                raise ValueError("Missing compressor state block-table entry")
            physical_block = state_block_table[logical_block]
            if not isinstance(physical_block, int) or not 0 < physical_block < state_cache.shape[0]:
                raise ValueError("Compressor state refers to an absent/null/out-of-range page")
            return state_cache[physical_block, offset]

        # Validate all pages needed by the chunk before changing any cache row.
        if x.shape[0]:
            first = start_pos - start_pos % self.ratio
            for position in range(first, start_pos + x.shape[0]):
                state_row(position)
        kv = self.wkv(x.float())
        score = self.wgate(x.float())
        completed = []
        for token in range(x.shape[0]):
            position = start_pos + token
            row = state_row(position)
            row[: self.width] = kv[token]
            row[self.width :] = score[token]
            if (position + 1) % self.ratio == 0:
                group = torch.stack([state_row(position - 1), row])
                pooled = (group[:, : self.width] * group[:, self.width :].softmax(dim=0)).sum(dim=0)
                completed.append(pooled)
        latent = torch.stack(completed).to(x.dtype) if completed else x.new_empty((0, self.width))
        return self.norm(latent)
