# DeepSeek V4.1 initialization bring-up

This is an initial **parameter/cache graph**, not a registered, runnable model.
Do not use this branch as a claim of checkpoint-loading or inference support.

## Implemented

- Validated backbone layer/source topology, including nested text configuration.
- Four cache groups: local SWA, ratio2 full, ratio1 full, compressor state SWA.
- Only source layers own long KV and Index K. Eight index selection layers do
  not imply eight Index K caches.
- Independent resource planes, exact byte accounting, startup memory checks,
  concurrency accounting, V1 collection/allocation/reshape/prefix binding.
- Model-owned cache modules with empty initial tensors; consumer lookup does
  not re-register a shared PyTorch module or allocate duplicate pages.
- Single-rank BF16 attention parameter graph, FP32 ratio2 gated compressor,
  ratio1 projection-only compressor, and owner-dependent Indexer parameters.
- Cache metadata and completed-group slot mapping. Compressor state keeps one
  FP32 KV/score row per original token, using ordinary SWA block tables and slots.

## Integration contract

The package follows V4's component ownership. Construct
`DeepseekV41Attention(text_config, layer_idx, vllm_config, prefix)` per layer:

- `model.py`: Attention owns local SWA and, at source layers, long KV.
- `compressor.py`: the ratio2 Compressor owns `state_cache`; ratio1 has none.
- `indexer.py`: source Indexers own `k_cache`; query-only Indexers have none.
- `attention/dsa_v41.py`: cache-layer registration, layout and metadata.
- `core/deepseek_v41.py`: CacheSpecs, hybrid grouping, sizing and allocation.

There is no model-side `kv_cache.py`, centralized cache owner, or cache-plan
construction pass. Each component registers its cache under its full prefix in
`static_forward_context`, like V4. Consumers retain source-prefix references
instead of registering duplicate cache modules. The runner collects specs from
these actual component-owned modules; the framework groups them afterward.

The first implementation requires model runner V1, eager, single rank,
hybrid cache management, and BF16 cache storage. It rejects prefix caching,
PD/KV transfer, speculative decoding and parallel configurations. Kernel-side
numerical FP4/FP8 rounding from the reference is not supplied by BF16 storage.

Compressor state uses `AscendSlidingWindowMLASpec` with window two and the
standard `SlidingWindowManager`, matching V4's state-cache management. Its
storage compression ratio is one: ratio2 applies to pooling, not raw state rows.
The state page size is independently accounted; V4's C4/C128 padding constants
are not copied. State remains a separate group from the local window-128 KV.

The compressor accepts `[pages, block_size, 2*head_dim]` FP32 state (the cache's
singleton head dimension squeezed) and a state block table, matching the V4
operator's cache interface. The unfused per-request reference uses host page IDs;
the future fused call will use the batched device block table. It overwrites
each token's KV/score row before pooling, so recycled pages need no request-wide
reset. Historical rows needed at a chunk boundary must remain in the SWA table.
This is a reference bring-up component, not a throughput optimization.

This change does not enable the existing compressor kernel for ratio2 or prove
its no-overlap/no-APE semantics. Operator adaptation and numerical validation
are still required before wiring that call.

## Still required

- Full model/config registration, backbone mHC/MoE/Engram/vision integration,
  checkpoint weight conversion/loading, and complete LM forward.
- RoPE and reference numerical transforms, Candidate/TopK generation and
  sharing, cache writes/gathers, sparse attention and attention-sink semantics.
- Request lifecycle validation for state retention/recompute, profile scratch budgeting,
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
state paging, recycled pages, metadata slots and ratio2 chunked pooling against
a vector reference.

At initial submission these tests are **added but not executed**; only static
syntax/lint/diff checks are available. No model, accuracy or performance pass
is claimed. Check the accompanying workspace validation report for omissions.
