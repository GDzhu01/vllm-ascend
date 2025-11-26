# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
import vllm.v1.sample.rejection_sampler as rs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (RejectionSampler, apply_sampling_constraints,
                                              generate_uniform_probs)
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

import triton
import triton.language as tl

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32


class AscendRejectionSampler(RejectionSampler, nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def forward(
            self,
            metadata: SpecDecodeMetadata,
            # [num_tokens, vocab_size]
            draft_probs: Optional[torch.Tensor],
            # [num_tokens, vocab_size]
            target_logits: torch.Tensor,
            # [batch_size, 1]
            bonus_token_ids: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            target_logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens, vocab_size]. Here, probabilities from
                different requests are flattened into a single tensor because
                this is the shape of the output logits.
                NOTE: `target_logits` can be updated in place to save memory.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1].
                Bonus tokens are added to the end of the sequence if all
                proposed tokens are accepted. We generate the bonus tokens
                outside of the rejection sampler with the default sampling
                strategy. It allows for more flexibility in the sampling
                process such as top_p, top_k sampling.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''
        assert metadata.max_spec_len <= MAX_SPEC_LEN
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `compute_probs` function.
        # 在all_greedy下不正常，会提前返回
        target_probs = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        target_probs = target_probs.softmax(dim=-1, dtype=torch.float32)
        # Switch of Block Verify: when MTP>=2, using block verify for rejection sampler.
        using_block_verify = metadata.max_spec_len >= 2
        # if using_block_verify and sampling_metadata.all_greedy:
        #     target_probs = compute_probs_for_block_verify(
        #         target_logits,
        #         metadata.cu_num_draft_tokens,
        #         sampling_metadata,
        #     )

        # print("target_probs have got!", target_probs)
        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_probs,
            bonus_token_ids,
            sampling_metadata,
            using_block_verify
        )
        return output_token_ids


def compute_probs_for_block_verify(
        logits: torch.Tensor,  # [num_tokens, vocab_size]
        cu_num_draft_tokens: torch.Tensor,  # [batch_size]
        sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Compute probability distribution from logits based on sampling metadata.

    This function applies temperature scaling to the logits and converts
    them to probabilities using softmax. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be converted to probabilities.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Probability distribution (softmax of scaled logits)
            if non-greedy sampling is used, otherwise returns the
            original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1

    num_tokens = logits.shape[0]

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    logits = apply_top_k_top_p(logits, top_k, top_p)
    output_prob = logits.softmax(dim=-1, dtype=torch.float32)
    return output_prob


def rejection_sample(
        # [num_tokens]
        draft_token_ids: torch.Tensor,
        # [batch_size]
        num_draft_tokens: list[int],
        max_spec_len: int,
        # [batch_size]
        cu_num_draft_tokens: torch.Tensor,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_probs: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        using_block_verify: bool = False
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not using_block_verify:
        if not sampling_metadata.all_random:
            # Rejection sampling for greedy sampling requests.
            target_argmax = target_probs.argmax(dim=-1)
            if min(num_draft_tokens) == 1 and max(
                    num_draft_tokens) == 1 and sampling_metadata.all_greedy:
                rejection_greedy_sample_spec_len_1_pytorch(
                    output_token_ids,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                )
            else:
                rejection_greedy_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    num_draft_tokens,
                    max_spec_len,
                    is_greedy,
                )
            if sampling_metadata.all_greedy:
                return output_token_ids

    # if not sampling_metadata.all_random:
    #     # Rejection sampling for greedy sampling requests.
    #     target_argmax = target_probs.argmax(dim=-1)
    #     rejection_greedy_sample_kernel[(batch_size,)](
    #         output_token_ids,
    #         cu_num_draft_tokens,
    #         draft_token_ids,
    #         target_argmax,
    #         bonus_token_ids,
    #         is_greedy,
    #         max_spec_len,
    #     )
    #     if sampling_metadata.all_greedy:
    #         return output_token_ids
    # print("\n----------------------------------------\n")
    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    # 从0~1均匀分布，采样，0.4
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    ).to(torch.float32)

    # recovered_token_ids = sample_recovered_tokens(
    #     max_spec_len,
    #     num_draft_tokens,
    #     cu_num_draft_tokens,
    #     draft_token_ids,
    #     draft_probs,
    #     target_probs,
    #     sampling_metadata,
    #     device,
    # )

    # Rejection sampling for random sampling requests.

    # rejection_random_sample_pytorch(
    #     output_token_ids,
    #     cu_num_draft_tokens,
    #     draft_token_ids,
    #     draft_probs,
    #     target_probs,
    #     bonus_token_ids,
    #     uniform_probs,
    #     is_greedy,
    #     max_spec_len,
    #     vocab_size,
    #     using_block_verify,
    #     IS_NGRAM=draft_probs is None,
    # )
    # Rejection sampling for random sampling requests.
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        # recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    # print("\n-----------------------------")
    # print("output_token_ids", output_token_ids)
    # print("-----------------------------\n")

    # rejection_random_sample_pytorch_Multiprocessing(
    #     output_token_ids,
    #     cu_num_draft_tokens,
    #     draft_token_ids,
    #     draft_probs,
    #     target_probs,
    #     bonus_token_ids,
    #     uniform_probs,
    #     is_greedy,
    #     max_spec_len,
    #     vocab_size,
    #     using_block_verify,
    #     IS_NGRAM=draft_probs is None,
    #     # num_warps=1,
    # )

    # rejection_random_sample_pytorch_CPU_loop(
    #     output_token_ids,
    #     cu_num_draft_tokens,
    #     draft_token_ids,
    #     draft_probs,
    #     target_probs,
    #     bonus_token_ids,
    #     uniform_probs,
    #     is_greedy,
    #     using_block_verify,
    #     IS_NGRAM=draft_probs is None,
    # )

    # rejection_random_sample_pytorch_NPU_loop(
    #     output_token_ids,
    #     cu_num_draft_tokens,
    #     draft_token_ids,
    #     draft_probs,
    #     target_probs,
    #     bonus_token_ids,
    #     uniform_probs,
    #     num_draft_tokens,
    #     is_greedy,
    #     using_block_verify,
    #     IS_NGRAM=draft_probs is None,
    # )

    return output_token_ids


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
        output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,  # [num_tokens]
        target_argmax_ptr,  # [num_tokens]
        bonus_token_ids_ptr,  # [batch_size]
        is_greedy_ptr,  # [batch_size] or None
        max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel_block_verify(
        output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,  # [num_tokens]
        target_argmax_ptr,  # [num_tokens]
        bonus_token_ids_ptr,  # [batch_size]
        is_greedy_ptr,  # [batch_size] or None
        max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    last_accepted_token_pos = -1

    for pos in range(num_draft_tokens):
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
        if draft_token_id == target_argmax_id:
            last_accepted_token_pos = pos
            rejected = False
        else:
            rejected = True

    if last_accepted_token_pos > -1:
        for pos in range(last_accepted_token_pos + 1):
            token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if rejected:
        recovered_id = tl.load(draft_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1, recovered_id
        )
    else:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
        output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens_ptr,  # [batch_size]
        draft_token_ids_ptr,  # [num_tokens]
        draft_probs_ptr,  # [num_tokens, vocab_size] or None
        target_probs_ptr,  # [num_tokens, vocab_size]
        bonus_token_ids_ptr,  # [batch_size]
        # recovered_token_ids_ptr,  # [num_tokens]
        uniform_probs_ptr,  # [num_tokens]
        is_greedy_ptr,  # [batch_size]
        max_spec_len,
        vocab_size,
        NO_DRAFT_PROBS: tl.constexpr,
        SUB_BLOCK: tl.constexpr = 1500,
):
    req_idx = tl.program_id(0)
    # is_greedy = tl.load(is_greedy_ptr + req_idx)
    # if is_greedy:
    #     # Early exit for greedy sampling requests.
    #     return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    pi = 1.0
    uniform_prob = 1.0
    last_accepted_token_pos = -1

    for pos in range(num_draft_tokens):
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        target_prob = tl.load(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
        )
        tmp_uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
        uniform_prob = uniform_prob * tmp_uniform_prob

        if NO_DRAFT_PROBS:
            draft_prob = 1
        else:
            draft_prob = tl.load(
                draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )

        pi = min(pi * target_prob / draft_prob, 1.0)
        if draft_prob > 0 and pi >= uniform_prob:
            last_accepted_token_pos = pos
            rejected = False
        else:
            rejected = True

    if last_accepted_token_pos > -1:
        for pos in range(last_accepted_token_pos + 1):
            token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )

    if rejected:
        loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        global_recovered_id = -1
        global_max_p = -1.0
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            tmp_target_prob = tl.load(
                target_probs_ptr + (start_idx + last_accepted_token_pos + 1) * vocab_size + vocab_offset,
                mask=vocab_offset < vocab_size,
                other=0
            )
            recovered_id = tl.argmax(tmp_target_prob, axis=-1)
            max_p = tl.get_element(tmp_target_prob, (recovered_id,))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id

        # vocab_offset = tl.arange(0, PADDED_VOCAB_SIZE)
        # tmp_target_prob = tl.load(
        #     target_probs_ptr + (start_idx + last_accepted_token_pos + 1) * vocab_size + vocab_offset
        # )
        # recovered_token_id = tl.argmax(tmp_target_prob, axis=-1)

        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1, global_recovered_id
        )
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        # print("bonus_token_id",bonus_token_id)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id
        )


def expand_batch_to_tokens(
        x: torch.Tensor,  # [batch_size]
        cu_num_tokens: torch.Tensor,  # [batch_size]
        num_tokens: int,
        replace_from: int = 0,
        replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_pytorch(
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def sample_recovered_tokens(
        max_spec_len: int,
        num_draft_tokens: list[int],
        # [batch_size]
        cu_num_draft_tokens: torch.Tensor,
        # [num_tokens]
        draft_token_ids: torch.Tensor,
        # [num_tokens, vocab_size]
        draft_probs: Optional[torch.Tensor],
        # [num_tokens, vocab_size]
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    sample_recovered_tokens_pytorch(
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        IS_NGRAM=draft_probs is None,
    )
    return recovered_token_ids


def rejection_greedy_sample_spec_len_1_pytorch(
        output_token_ids,  # [batch_size, 2]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    assert batch_size == num_tokens
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[accept_req_mask, 1] = bonus_token_ids[accept_req_mask]


def rejection_greedy_sample_pytorch(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        max_spec_len,
        is_greedy=None,  # [batch_size] or None
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device

    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(
        device, non_blocking=True)

    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(
        num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    mismatch_global = (draft_token_ids != target_argmax)
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size,
                                                 dtype=torch.long,
                                                 device=device)
    else:
        # [bs, max_spec_len]
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len),
                                     False,
                                     dtype=torch.bool,
                                     device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix,
                                         max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = (first_mismatch_pos_per_req == max_spec_len * 2)
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[
            no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1,
                             draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1,
                                device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[
        global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req
                               >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]


def rejection_random_sample_pytorch(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        # recovered_token_ids,  # [num_tokens]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        max_spec_len,
        vocab_size,
        using_block_verify,  # bool
        IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]

    for req_idx in range(batch_size):
        if not using_block_verify and is_greedy is not None and is_greedy[req_idx]:
            continue
        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        num_draft_tokens = end_idx - start_idx

        rejected = False
        pi = 1.0
        uniform_prob = 1.0
        last_accepted_token_pos = -1
        for pos in range(num_draft_tokens):
            draft_token_id = draft_token_ids[start_idx + pos].item()

            target_prob = target_probs[start_idx + pos, draft_token_id].item()
            uniform_prob = uniform_prob * uniform_probs[start_idx + pos].item()

            if IS_NGRAM:
                draft_prob = 1.0
            else:
                draft_prob = draft_probs[start_idx + pos, draft_token_id].item()

            pi = min(pi * target_prob / draft_prob, 1.0)

            # TODO: check weather we need h.

            if draft_prob > 0 and pi >= uniform_prob:
                last_accepted_token_pos = pos
                rejected = False
            else:
                rejected = True

        if last_accepted_token_pos > -1:
            for pos in range(last_accepted_token_pos + 1):
                draft_token_id = draft_token_ids[start_idx + pos].item()
                output_token_ids[req_idx, pos] = draft_token_id

        if rejected:
            # recovered_token_id = recovered_token_ids[start_idx + last_accepted_token_pos + 1].item()
            recovered_token_id = torch.argmax(target_probs[start_idx + last_accepted_token_pos + 1]).item()
            output_token_ids[req_idx, last_accepted_token_pos + 1] = recovered_token_id
        else:
            bonus_token_id = bonus_token_ids[req_idx].item()
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_id

        # print("num_draft_tokens:", num_draft_tokens, "tokens,", "accept_tokens:", last_accepted_token_pos + 1, "tokens")
        # print("output_token_ids:", output_token_ids[req_idx])
        # print('\n')


def expand_pytorch(
        output_ptr,  # [num_tokens]
        input_ptr,  # [batch_size]
        cu_num_tokens_ptr,  # [batch_size]
        replace_from,
        replace_to,
        MAX_NUM_TOKENS,
):
    batch_size = len(input_ptr)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_tokens_ptr[req_idx - 1]
        end_idx = cu_num_tokens_ptr[req_idx]
        num_tokens = end_idx - start_idx

        src_val = input_ptr[req_idx]
        src_val = replace_to if src_val == replace_from else src_val

        offset = torch.arange(MAX_NUM_TOKENS, device=num_tokens.device)
        mask = offset < num_tokens

        output_slice = start_idx + offset[mask]
        output_ptr[output_slice] = src_val


def sample_recovered_tokens_pytorch(
        output_token_ids,  # [num_tokens]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        q,  # [batch_size, vocab_size]
        vocab_size,
        IS_NGRAM=False,
):
    batch_size = len(cu_num_draft_tokens)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1]
        end_idx = cu_num_draft_tokens[req_idx]
        num_draft_tokens = end_idx - start_idx

        for pos in range(num_draft_tokens):
            token_idx = start_idx + pos

            if IS_NGRAM:
                draft_token_id = draft_token_ids[token_idx]
                orig_prob = target_probs[token_idx, draft_token_id].item()
                target_probs[token_idx, draft_token_id] = 0
                prob = target_probs[token_idx].clone()
            else:
                draft_p = draft_probs[token_idx].clone()
                target_p = target_probs[token_idx].clone()
                prob = torch.maximum(target_p - draft_p,
                                     torch.tensor(0.0, device=target_p.device))

            q_values = torch.full((vocab_size,),
                                  float('-inf'),
                                  device=q.device)
            q_values[:vocab_size] = q[req_idx, :vocab_size]

            recovered_id = torch.argmax(prob / q_values).item()
            output_token_ids[token_idx] = recovered_id

            if IS_NGRAM:
                target_probs[token_idx, draft_token_id] = orig_prob


def rejection_random_sample_pytorch_CPU_loop(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        using_block_verify,  # bool
        IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]

    cu_num_draft_tokens_cpu = cu_num_draft_tokens.cpu()
    draft_token_ids_cpu = draft_token_ids.cpu()
    bonus_token_ids_cpu = bonus_token_ids.cpu()

    target_probs_cpu = target_probs.cpu()
    uniform_probs_cpu = uniform_probs.cpu()

    if not IS_NGRAM:
        draft_probs_cpu = draft_probs.cpu() if draft_probs is not None else None
    else:
        draft_probs_cpu = None

    if is_greedy is not None:
        is_greedy_cpu = is_greedy.cpu()
    else:
        is_greedy_cpu = None

    for req_idx in range(batch_size):
        if not using_block_verify and is_greedy_cpu is not None and is_greedy_cpu[req_idx]:
            continue

        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = cu_num_draft_tokens_cpu[req_idx - 1]
        end_idx = cu_num_draft_tokens_cpu[req_idx]
        num_draft_tokens = end_idx - start_idx

        if num_draft_tokens == 0:
            bonus_token_id = bonus_token_ids_cpu[req_idx]
            output_token_ids[req_idx, 0] = bonus_token_id
            continue

        rejected = False
        pi = 1.0
        uniform_prob = 0.1
        last_accepted_token_pos = -1

        # 处理每个draft token
        for pos in range(num_draft_tokens):
            draft_token_id = draft_token_ids_cpu[start_idx + pos]
            target_prob = target_probs_cpu[start_idx + pos, draft_token_id]
            uniform_prob = uniform_prob * uniform_probs_cpu[start_idx + pos]

            if IS_NGRAM:
                draft_prob = 1.0
            else:
                draft_prob = draft_probs_cpu[start_idx + pos, draft_token_id]

            pi = min(pi * target_prob / draft_prob, 1.0) if draft_prob > 0 else 0.0

            if draft_prob > 0 and pi >= uniform_prob:
                last_accepted_token_pos = pos
                rejected = False
            else:
                rejected = True

        if last_accepted_token_pos >= 0:
            accepted_tokens = draft_token_ids_cpu[start_idx:start_idx + last_accepted_token_pos + 1]
            for pos, token_id in enumerate(accepted_tokens):
                output_token_ids[req_idx, pos] = token_id

        if rejected:
            # target_probs_slice = target_probs_cpu[start_idx + last_accepted_token_pos + 1]
            target_probs_slice = target_probs[start_idx + last_accepted_token_pos + 1]
            recovered_token_id = torch.argmax(target_probs_slice).item()
            output_token_ids[req_idx, last_accepted_token_pos + 1] = recovered_token_id
        else:
            bonus_token_id = bonus_token_ids_cpu[req_idx]
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_id


def rejection_random_sample_pytorch_CPU_matrix(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        # recovered_token_ids,  # [num_tokens]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        max_spec_len,
        vocab_size,
        IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]
    num_tokens = draft_token_ids.size(0)

    device = 'cpu'
    # device_bak = output_token_ids.device
    # output_token_ids = output_token_ids.cpu()
    cu_num_draft_tokens_cpu = cu_num_draft_tokens.cpu()
    draft_token_ids_cpu = draft_token_ids.cpu()
    target_probs_cpu = target_probs.cpu()
    bonus_token_ids_cpu = bonus_token_ids.cpu()
    uniform_probs_cpu = uniform_probs.cpu()

    target_probs_draft = target_probs_cpu[torch.arange(len(draft_token_ids_cpu)), draft_token_ids_cpu]
    # target_probs_draft = torch.tensor([1, 1, 1, 1, 0.1, 1, 0.1, 1, 0, 0.1, 1, 0])

    draft_tokens_per_req_cpu = torch.tensor(draft_tokens_per_req)
    start_indices = cu_num_draft_tokens_cpu - draft_tokens_per_req_cpu
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req_cpu)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    p_uniform = 1.0

    for idx in range(num_tokens):
        if idx == start_indices[token_req_ids[idx]]:
            uniform_probs_cpu[idx] *= p_uniform
        else:
            target_probs_draft[idx] = min(target_probs_draft[idx - 1] * target_probs_draft[idx], 1.0)
            uniform_probs_cpu[idx] *= uniform_probs_cpu[idx - 1]

    if IS_NGRAM:
        draft_probs = 1.0
    else:
        for idx in range(num_tokens):
            if idx != start_indices[token_req_ids[idx]]:
                draft_probs[idx] = min(draft_probs[idx - 1] * draft_probs[idx], 1.0)

    accept_global = (target_probs_draft / draft_probs >= uniform_probs_cpu)

    if max_spec_len == 0:
        last_accept_pos_per_req = torch.full(batch_size,
                                             -1,
                                             dtype=torch.long,
                                             device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        accept_matrix = torch.full((batch_size, max_spec_len),
                                   False,
                                   dtype=torch.bool,
                                   device=device)
        accept_matrix[token_req_ids, token_positions] = accept_global
        accept_positions = torch.where(accept_matrix, pos_matrix, -1)
        last_accept_pos_per_req, _ = torch.max(accept_positions, dim=1)

    copy_len = torch.minimum(last_accept_pos_per_req + 1, draft_tokens_per_req_cpu)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[copy_mask] = draft_token_ids_cpu[global_idx[copy_mask].to(output_token_ids.dtype)].to(
        output_token_ids.device)

    needs_recover = last_accept_pos_per_req + 1 < draft_tokens_per_req_cpu

    if torch.any(needs_recover):
        recover_rows = torch.where(needs_recover)[0]
        recover_cols = last_accept_pos_per_req[recover_rows] + 1
        output_token_ids[recover_rows, recover_cols] = torch.argmax(
            target_probs_cpu[start_indices[recover_rows] + recover_cols], dim=-1).to(output_token_ids.dtype).to(
            output_token_ids.device)

    needs_bonus = last_accept_pos_per_req + 1 >= draft_tokens_per_req_cpu

    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req_cpu[bonus_rows]
        bonus_token_ids_cpu = bonus_token_ids_cpu.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids_cpu[bonus_rows].to(output_token_ids.device)
    return output_token_ids


def rejection_random_sample_pytorch_NPU_loop(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        uniform_probs,  # [num_tokens]
        draft_tokens_per_req,  # [batch_size], list
        is_greedy,  # [batch_size]
        using_block_verify,  # bool
        IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(
        device, non_blocking=True)

    if not using_block_verify and is_greedy is not None:
        skip_mask = is_greedy
    else:
        skip_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    for req_idx in range(batch_size):
        if skip_mask[req_idx]:
            continue
        start_idx = start_indices[req_idx]
        num_draft_tokens = draft_tokens_per_req[req_idx]

        if num_draft_tokens == 0:
            output_token_ids[req_idx, 0] = bonus_token_ids[req_idx]
            continue

        req_draft_tokens = draft_token_ids[start_idx: start_idx + num_draft_tokens]
        req_target_probs = target_probs[start_idx: start_idx + num_draft_tokens]
        req_uniform_probs = uniform_probs[start_idx: start_idx + num_draft_tokens]
        if IS_NGRAM:
            req_draft_probs = torch.ones_like(req_draft_tokens, dtype=torch.float32)
        else:
            req_draft_probs = draft_probs[start_idx: start_idx + num_draft_tokens].gather(
                1, req_draft_tokens.unsqueeze(1)
            ).squeeze(1)
        target_probs_for_draft = req_target_probs.gather(
            1, req_draft_tokens.unsqueeze(1)
        ).squeeze(1)

        ratio = target_probs_for_draft / req_draft_probs.clamp(min=1e-8)
        cum_ratio = torch.cumprod(ratio, dim=0)
        pi_values = torch.minimum(cum_ratio, torch.ones_like(cum_ratio))
        cum_uniform_probs = torch.cumprod(req_uniform_probs, dim=0)

        acceptance_mask = (req_draft_probs > 0) & (pi_values >= cum_uniform_probs)

        accepted_positions = torch.where(acceptance_mask)[0]
        if len(accepted_positions) > 0:
            last_accepted_pos = accepted_positions[-1]
            if last_accepted_pos == num_draft_tokens - 1:
                rejected = False
            else:
                rejected = True
        else:
            last_accepted_pos = -1
            rejected = True

        if last_accepted_pos >= 0:
            accepted_tokens = req_draft_tokens[:last_accepted_pos + 1]
            output_token_ids[req_idx, :last_accepted_pos + 1] = accepted_tokens

        if rejected:
            recovered_token = torch.argmax(req_target_probs[last_accepted_pos + 1])
            output_token_ids[req_idx, last_accepted_pos + 1] = recovered_token
        else:
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[req_idx]


def rejection_random_sample_pytorch_NPU_matrix(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        # recovered_token_ids,  # [num_tokens]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        max_spec_len,
        vocab_size,
        IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device

    target_probs_draft = target_probs[torch.arange(len(draft_token_ids)), draft_token_ids]
    # target_probs_draft = torch.tensor([1, 1, 1, 1, 0.1, 1, 0.1, 1, 0, 0.1, 1, 0])

    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(device, non_blocking=True)
    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    p_uniform = 1.0

    for idx in range(num_tokens):
        if idx == start_indices[token_req_ids[idx]]:
            uniform_probs[idx] *= p_uniform
        else:
            target_probs_draft[idx] = min(target_probs_draft[idx - 1] * target_probs_draft[idx], 1.0)
            uniform_probs[idx] *= uniform_probs[idx - 1]

    if IS_NGRAM:
        draft_probs = 1.0
    else:
        for idx in range(num_tokens):
            if idx != start_indices[token_req_ids[idx]]:
                draft_probs[idx] = min(draft_probs[idx - 1] * draft_probs[idx], 1.0)

    accept_global = (target_probs_draft / draft_probs >= uniform_probs)

    if max_spec_len == 0:
        last_accept_pos_per_req = torch.full(batch_size,
                                             -1,
                                             dtype=torch.long,
                                             device=device)
    else:
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        accept_matrix = torch.full((batch_size, max_spec_len),
                                   False,
                                   dtype=torch.bool,
                                   device=device)
        accept_matrix[token_req_ids, token_positions] = accept_global
        accept_positions = torch.where(accept_matrix, pos_matrix, -1)
        last_accept_pos_per_req, _ = torch.max(accept_positions, dim=1)

    copy_len = torch.minimum(last_accept_pos_per_req + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[copy_mask] = draft_token_ids[global_idx[copy_mask].to(output_token_ids.dtype)]

    needs_recover = last_accept_pos_per_req + 1 < draft_tokens_per_req

    if torch.any(needs_recover):
        recover_rows = torch.where(needs_recover)[0]
        recover_cols = last_accept_pos_per_req[recover_rows] + 1
        output_token_ids[recover_rows, recover_cols] = torch.argmax(
            target_probs[start_indices[recover_rows] + recover_cols], dim=-1).to(output_token_ids.dtype)

    needs_bonus = last_accept_pos_per_req + 1 >= draft_tokens_per_req

    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]

    return output_token_ids


import multiprocessing as mp


def rejection_random_sample_pytorch_Multiprocessing(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        # recovered_token_ids,  # [num_tokens]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        max_spec_len,
        vocab_size,
        using_block_verify,  # bool
        IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]
    cu_num_draft_tokens_cpu = cu_num_draft_tokens.cpu()
    draft_token_ids_cpu = draft_token_ids.cpu()
    draft_probs_cpu = draft_probs.cpu() if draft_probs is not None else draft_probs
    target_probs_cpu = target_probs.cpu()
    bonus_token_ids_cpu = bonus_token_ids.cpu()
    uniform_probs_cpu = uniform_probs.cpu()

    processes = []

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    for req_idx in range(batch_size):
        p = ctx.Process(target=process_request, args=(req_idx,
                                                      cu_num_draft_tokens_cpu,  # [batch_size]
                                                      draft_token_ids_cpu,  # [num_tokens]
                                                      draft_probs_cpu,  # [num_tokens, vocab_size] or None
                                                      target_probs_cpu,  # [num_tokens, vocab_size]
                                                      bonus_token_ids_cpu,  # [batch_size]
                                                      uniform_probs_cpu,  # [num_tokens]
                                                      is_greedy,  # [batch_size]
                                                      max_spec_len,
                                                      vocab_size,
                                                      using_block_verify,  # bool
                                                      IS_NGRAM,
                                                      result_queue)
                        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    while not result_queue.empty():
        try:
            req_idx, result = result_queue.get_nowait()
            for pos, token_id in result:
                output_token_ids[req_idx, pos] = token_id
        except:
            break

    return output_token_ids


def process_request(
        req_idx,
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        draft_probs,  # [num_tokens, vocab_size] or None
        target_probs,  # [num_tokens, vocab_size]
        bonus_token_ids,  # [batch_size]
        # recovered_token_ids,  # [num_tokens]
        uniform_probs,  # [num_tokens]
        is_greedy,  # [batch_size]
        max_spec_len,
        vocab_size,
        using_block_verify,  # bool
        IS_NGRAM,
        result_queue):
    if not using_block_verify and is_greedy is not None and is_greedy[req_idx]:
        return

    start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1].item()
    end_idx = cu_num_draft_tokens[req_idx].item()
    num_draft_tokens = end_idx - start_idx
    rejected = False
    token_results = []

    pi = 1.0
    uniform_prob = 1.0
    last_accepted_token_pos = -1

    for pos in range(num_draft_tokens):
        draft_token_id = draft_token_ids[start_idx + pos].item()

        target_prob = target_probs[start_idx + pos, draft_token_id].item()
        uniform_prob = uniform_prob * uniform_probs[start_idx + pos].item()

        if IS_NGRAM:
            draft_prob = 1.0
        else:
            draft_prob = draft_probs[start_idx + pos, draft_token_id].item()

        pi = min(pi * target_prob / draft_prob, 1.0)

        # TODO: check weather we need h.

        if draft_prob > 0 and pi >= uniform_prob:
            last_accepted_token_pos = pos
            rejected = False
        else:
            rejected = True

    if last_accepted_token_pos > -1:
        for pos in range(last_accepted_token_pos + 1):
            draft_token_id = draft_token_ids[start_idx + pos].item()
            token_results.append((pos, draft_token_id))

    if rejected:
        recovered_token_id = torch.argmax(target_probs[start_idx + last_accepted_token_pos + 1]).item()
        token_results.append((last_accepted_token_pos + 1, recovered_token_id))
    else:
        bonus_token_id = bonus_token_ids[req_idx].item()
        token_results.append((num_draft_tokens, bonus_token_id))
    result_queue.put((req_idx, token_results))


rs.expand_batch_to_tokens = expand_batch_to_tokens
