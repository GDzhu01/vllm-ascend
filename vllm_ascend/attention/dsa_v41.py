# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 cache metadata. Sparse-attention execution is a separate next step."""

from dataclasses import dataclass

import torch
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata, AttentionMetadataBuilder

from vllm_ascend.models.deepseek_v41.kv_cache import DeepseekV41TailSpec


@dataclass
class DeepseekV41Metadata(AttentionMetadata):
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_tail: bool


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    if ratio not in (1, 2):
        raise ValueError("V4.1 only supports ratio 1 or 2")
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def tail_slot_mapping(block_table, query_start_loc, seq_lens, num_tokens):
    """Fixed request block plus absolute position modulo two; mask padded tokens."""
    if seq_lens.shape[0] == 0:
        return torch.full((num_tokens,), -1, dtype=torch.int64, device=block_table.device)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    token_ids = torch.arange(num_tokens, device=query_start_loc.device)
    request_ids = torch.bucketize(token_ids, query_start_loc[1:], right=True)
    safe_requests = request_ids.clamp(max=seq_lens.shape[0] - 1)
    positions = seq_lens[safe_requests] - query_lens[safe_requests]
    positions = positions + token_ids - query_start_loc[safe_requests]
    blocks = block_table[safe_requests, 0]
    valid = (request_ids < seq_lens.shape[0]) & (blocks > 0)
    return torch.where(valid, blocks.to(torch.int64) * 2 + positions % 2, -1)


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder[DeepseekV41Metadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len:
            raise NotImplementedError("V4.1 prefix caching is not implemented")
        spec = self.kv_cache_spec
        common = common_attn_metadata
        is_tail = isinstance(spec, DeepseekV41TailSpec)
        ratio = getattr(spec, "compress_ratio", 1)
        # Tail uses block_table[:, 0] and absolute position modulo two. Ordinary
        # slot_mapping cannot index its fixed block after the first token block.
        slots = (
            tail_slot_mapping(
                common.block_table_tensor, common.query_start_loc, common.seq_lens, common.slot_mapping.shape[0]
            )
            if is_tail
            else compressed_slot_mapping(common.slot_mapping, ratio)
        )
        return DeepseekV41Metadata(
            common.block_table_tensor,
            common.query_start_loc,
            common.seq_lens,
            slots,
            ratio,
            spec.storage_block_size,
            is_tail,
        )


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("V4.1 sparse-attention execution is not implemented yet")

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size
