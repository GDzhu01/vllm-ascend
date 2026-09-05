# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 Indexer parameters; only KV source layers own index keys."""

import torch
from torch import nn

from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.core.deepseek_v41 import DeepseekV41FullSpec

from .compressor import DeepseekV41RMSNorm, _read


class DeepseekV41Indexer(nn.Module):
    def __init__(self, config, owns_k, vllm_config, prefix, compress_ratio):
        super().__init__()
        self.owns_k = owns_k
        heads = _read(config, "index_n_heads")
        width = _read(config, "index_head_dim")
        self.wq_b = nn.Linear(_read(config, "q_lora_rank"), heads * width, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(_read(config, "hidden_size"), heads, bias=False, dtype=torch.bfloat16)
        if owns_k:
            self.wk = nn.Linear(_read(config, "head_dim"), width, bias=False, dtype=torch.bfloat16)
            self.k_norm = DeepseekV41RMSNorm(width, _read(config, "rms_norm_eps"))
            self.k_cache = DeepseekV41CacheLayer(
                vllm_config,
                f"{prefix}.k_cache",
                DeepseekV41FullSpec(
                    block_size=vllm_config.cache_config.block_size,
                    num_kv_heads=1,
                    head_size=width,
                    dtype=torch.bfloat16,
                    compress_ratio=compress_ratio,
                ),
            )
