# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 Indexer parameters; only KV source layers own index keys."""

import torch
from torch import nn

from .compressor import DeepseekV41RMSNorm
from .kv_cache import _read


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
