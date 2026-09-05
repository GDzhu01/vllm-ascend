# DeepSeek V4.1 initialization bring-up

This is an initial **parameter/cache graph**, not a registered, runnable model.
Do not use this branch as a claim of checkpoint-loading or inference support.

## Implemented

- Validated backbone layer/source topology, including nested text configuration.
- Four cache groups: SWA, ratio2 full, ratio1 full, request-private compressor tail.
- Only source layers own long KV and Index K. Eight index selection layers do
  not imply eight Index K caches.
- Independent resource planes, exact byte accounting, startup memory checks,
  concurrency accounting, V1 collection/allocation/reshape/prefix binding.
- Model-owned cache modules with empty initial tensors; consumer lookup does
  not re-register a shared PyTorch module or allocate duplicate pages.
- Single-rank BF16 attention parameter graph, FP32 ratio2 gated compressor,
  ratio1 projection-only compressor, and owner-dependent Indexer parameters.
- Cache metadata and completed-group slot mapping. Tail uses its request block
  and absolute token position modulo two, not ordinary token slot mapping.

## Integration contract

Attach `DeepseekV41ModelCaches(vllm_config, prefix)` **once** to the model.
Construct `DeepseekV41Attention(text_config, layer_idx, caches.plan)` per layer.
Cache owners are registered under complete attention resource prefixes in
`static_forward_context`; attention modules retain only those prefix strings.

The first implementation requires model runner V1, eager, single rank,
hybrid cache management, and BF16 cache storage. It rejects prefix caching,
PD/KV transfer, speculative decoding and parallel configurations. Kernel-side
numerical FP4/FP8 rounding from the reference is not supplied by BF16 storage.

The model runner initializes raw cache memory, but request admission/recompute
must call `reset_tail` before first use. Tail reset is not a per-forward action.
The compressor accepts a scheduler-selected state view and supports arbitrary
chunk boundaries for one request. This unfused implementation is a reference
bring-up component, not a throughput optimization.

## Still required

- Full model/config registration, backbone mHC/MoE/Engram/vision integration,
  checkpoint weight conversion/loading, and complete LM forward.
- RoPE and reference numerical transforms, Candidate/TopK generation and
  sharing, cache writes/gathers, sparse attention and attention-sink semantics.
- Request lifecycle wiring for tail reset/recompute, profile scratch budgeting,
  and end-to-end engine startup on Ascend.
- Packed memory reuse, tensor parallel parameter loading, runner V2, graph,
  prefix, PD and speculative rollback. No support is inherited from V4.

The cache backend is cache-only; its implementation lookup and the unfinished
attention forward raise explicitly. No zero-output or dense substitute is used
to advertise inference readiness.

## Validation

Target tests (run in a remote Ascend development container, not the local PC):

```bash
pytest -q tests/ut/models/test_deepseek_v41_layer_plan.py \
  tests/ut/models/test_deepseek_v41_cache.py
```

Tests cover source/consumer ownership, resource counts, group/page accounting,
aliasing and reshape, unsupported modes, parameter ownership, slot boundaries,
tail reset isolation and ratio2 chunked pooling against a vector reference.

At initial submission these tests are **added but not executed**; only static
syntax/lint/diff checks are available. No model, accuracy or performance pass
is claimed. Check the accompanying workspace validation report for omissions.
