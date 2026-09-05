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
