# Sprint 117 - F8 Row2 Attribution and Next Fusion Target

Date: 2026-05-21

## Objective

Stop guessing about the remaining served `arena_f8_e4m3_b128_matmul_rows2`
bucket after Sprint 116. Add low-overhead, opt-in shape attribution for F8
arena matmul wrappers, profile the current default appliance, then use the
largest confirmed shape/callsite as the next production fusion target.

## Context

Sprint 116 shipped batched attention-projection F8 HMMA as a small default win,
but the current default is still only about `33.7` generated tok/s at
8-slot/256K. A Sprint 117 default profile was captured after the promotion:

| Bucket | GPU time | Calls | Avg |
|---|---:|---:|---:|
| `arena_f8_e4m3_b128_matmul_rows2_kernel` | 41.67% | 12,341 | 65.821 us |
| TurboMind SM70 MXFP4 grouped GEMM | 20.71% | 3,526 | 114.51 us |
| DS4 grouped attention-output-A rows2 | 12.94% | 1,763 | 143.10 us |
| F32 matmul | 4.90% | 3,526 | 27.09 us |
| `rms_norm_plain_kernel` | 3.20% | 1,763 | 35.37 us |
| `attention_decode_mixed_kernel` | 3.11% | 1,763 | 34.36 us |

The served profile did not visibly list the Sprint 116 HMMA attention-projection
kernels, while a direct full-scheduler `nvprof` run confirmed those kernels are
reachable. That makes the next step attribution, not another blind projection
kernel.

Artifacts:

- `logs/from-cluster/sprint117-default-profile/`

## Implementation Plan

1. Add `DS4_CUDA_F8_TRACE_SHAPES=1` instrumentation to the CUDA F8 arena matmul
   wrappers.
2. Trace at wrapper dispatch level, including wrapper kind, selected kernel path,
   GPU, rows, cols, token count, grouped shape, and call count.
3. Keep tracing off by default and avoid CUDA synchronization or device work.
4. Run the current default served appliance with the trace enabled.
5. Pick the largest remaining shape that is both hot and safe to batch/fuse.
6. Implement that production target behind a rollback flag, then run correctness
   and served A/B.

## Candidate Targets

Likely candidates before trace confirmation:

| Candidate | Expected shape | Why plausible |
|---|---:|---|
| attention `output_b` | `4096 x 8192` | Still called per slot from `grouped_attention_output` after batched attention projections |
| indexer q | `? x ?` | Ratio-4 layers can add unbatched F8 projection work at long context |
| shared FFN fallback | `2048 x 4096`, `4096 x 2048` | Should be batched in normal 4/8-slot mode, but trace should verify |
| other ungrouped F8 path | shape-dependent | The rows2 count is too large to optimize safely without attribution |

## Findings

The served default is not currently exercising the broad layer-batched HMMA
paths in the fast path. `async_pipeline_mode=per-step` pipelines slots across
GPU stages, but each stage processes a slot span of one. That keeps stage
overlap high, but it means the hot served path mostly dispatches per-slot F8
row-pair kernels.

Trace artifacts:

- `logs/from-cluster/sprint117-f8-shape-trace/`
- `logs/from-cluster/sprint117-sync-batch-probe/`
- `logs/from-cluster/sprint117-async-chunk4/`
- `logs/from-cluster/sprint117-single-pair-swiglu/`
- `logs/from-cluster/sprint117-single-pair-swiglu-trace/`

The wrapper trace uses power-of-two call logging, so the counts below are
summed per-GPU maxima and should be treated as attribution, not exact totals.

Default per-step trace at 8-slot/256K:

| Approx calls | Wrapper/path | Shape | Meaning |
|---:|---|---|---|
| `15360` | plain `rows2` | `2048 x 4096` | shared FFN gate and up, separately launched |
| `7680` | plain `rows2` | `1024 x 4096` | attention `q_a` projection |
| `7680` | plain `rows2` | `32768 x 1024` | attention `q_b` projection |
| `7680` | plain `rows2` | `512 x 4096` | attention `kv_latent` projection |
| `7680` | grouped `rows2_ds4_attn_o` | `8192 x 4096` | attention output A |
| `7680` | plain `rows2` | `4096 x 8192` | attention output B |
| `7680` | plain `rows2` | `4096 x 2048` | shared FFN down |

Sync batch probe:

- The non-async path does activate the Sprint 115/116 HMMA batch kernels.
- It is much slower for served use because it gives up per-step stage overlap.
- Representative response timing was `1.134382` generated tok/s/request; the
  soak harness did not emit an aggregate summary because it currently expects
  async timing fields for timed requests.

Chunked per-step probe:

- `DS4_V100_ASYNC_SLOT_CHUNK=4` is correct, but slow:
  `11.483646` generated tok/s at 8-slot/256K with `8/8` token match.
- The chunking path should remain diagnostic only. It recovers some batching
  shape but loses too much stage-pipeline granularity.

Single-slot shared gate/up/SwiGLU fusion:

- Added `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1`.
- The fused scalar wrapper replaces the two separate `2048 x 4096` shared
  gate/up matmuls and standalone SwiGLU with one `pair_swiglu_single` launch.
- Correctness passed:
  - `cuda_source_dtypes_smoke`
  - `cuda_v100_full_scheduler_smoke --slots 8 --expect-tm-layers 43`
  - `cuda_v100_selected_token_smoke --expected-token-hex 3136`
- Served result was neutral/slightly below default:
  `33.562643` generated tok/s at 8-slot/256K with `8/8` token match.
- Trace with the flag confirms the replacement:

| Approx calls | Wrapper/path | Shape |
|---:|---|---|
| `7680` | `pair_swiglu_single` scalar | `2048 x 4096` |
| `7680` | plain `rows2` | `4096 x 2048` |
| `7680` each | remaining attention/plain rows2 paths | unchanged |

Decision: keep the single-slot shared pair-SwiGLU path opt-in/off. It reduces
launches and intermediate traffic but does not raise appliance throughput
because it is still a scalar F8 row kernel. A useful fusion win here needs a
software-pipelined/Tensor-Core design that overlaps packed-weight decode,
activation staging, MMA, and epilogue work.

## Definition of Done

- [x] `DS4_CUDA_F8_TRACE_SHAPES=1` traces F8 wrapper dispatch shapes without
      changing default runtime behavior.
- [x] Current default 8-slot/256K served trace is captured on the V100 node.
- [x] Trace artifacts are copied under `logs/from-cluster/`.
- [x] The top remaining F8 row2 shapes/calls are summarized in this document and
      `docs/sprints/EXPERIMENT-STATUS.md`.
- [x] One traced hot target is implemented behind a rollback flag.
- [x] Source-format smoke, full scheduler, and selected-token oracle pass.
- [x] 8-slot/256K served A/B is measured with full token match.
- [ ] 4-slot/1M sanity is deferred because the opt-in fused scalar path was not
      promoted.
