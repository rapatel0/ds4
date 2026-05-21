# Sprint 122 - 16-Slot Profile And Rendezvous Stabilization

Date: 2026-05-21

## Objective

Profile the new 16-slot/256K throughput mode, test whether wider HMMA and
batched attention-output shapes can move the served path, and stabilize the
operator launcher so 16 concurrent requests reliably coalesce into one tensor
batch.

## Changes

- Admitted `n_tokens=16` for the guarded F8 HMMA shared-down,
  gate/up-SwiGLU, and attention-projection batch paths.
- Added a guarded 16-slot attention output-B batch probe behind
  `DS4_V100_BATCH_ATTN_OUTPUT_B=1`.
- Made `DS4_V100_ASYNC_SLOT_CHUNK` compatible with event handoff and records
  per-slot ready events for chunked workers.
- Raised the HTTP server backlog to at least the active microbatch.
- Changed launcher `DS4_V100_MICROBATCH_WAIT_US=auto` to resolve to `200000`
  us when `DS4_V100_ACTIVE_MICROBATCH >= 16`, while keeping the prior `50000`
  us default for smaller multi-slot serving.

## Validation

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_f8_hmma_attn_batch_smoke \
  tests/cuda_f8_hmma_pair_swiglu_smoke \
  tests/cuda_f8_hmma_shared_down_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tools/ds4-v100-replay
```

Cluster tests passed:

```text
tests/cuda_f8_hmma_attn_batch_smoke
tests/cuda_f8_hmma_pair_swiglu_smoke
tests/cuda_f8_hmma_shared_down_smoke
tests/cuda_v100_stage_scheduler_smoke --slots 16
tests/cuda_v100_full_scheduler_smoke --slots 16
```

All served A/B runs below preserved `16/16` token matches and zero request
errors.

## Results

| Case | Generated tok/s | Continuation tok/s | Decision |
|---|---:|---:|---|
| Candidate 16-token HMMA admission | `43.730215` | `40.997076` | Correct; best observed |
| Production auto after 200 ms rendezvous | `43.534061` | `40.813182` | Shipped default policy |
| Previous auto behavior, split into two 8-request batches | `34.730037` | `32.559410` | Fixed by auto wait |
| Explicit 200 ms wait | `43.635707` | `40.908475` | Confirms coalescing fix |
| Batch attention disabled | `43.617663` | `40.891559` | Neutral |
| Output-B batch probe | `43.625519` | `40.898924` | Keep opt-in/off |
| Pair-SwiGLU HMMA disabled | `43.614987` | `40.889051` | Neutral |
| Shared-down HMMA enabled | `34.837487` | `32.660144` | Keep opt-in/off |
| Async chunk 2, event handoff off | `39.487106` | `37.019162` | Keep opt-in/off |
| Async chunk 2, event handoff on | `28.876459` | `27.071680` | Keep opt-in/off |
| Async chunk 4, event handoff off | `17.811997` | `16.698747` | Keep opt-in/off |
| Async chunk 4, event handoff on | `18.447169` | `17.294221` | Keep opt-in/off |
| Async chunk 16, event handoff off | `13.315378` | `12.483167` | Keep opt-in/off |

The final production-auto run reported one coalesced tensor batch:

```text
ds4_v100_tensor_batched_groups_total 1
ds4_v100_tensor_batched_requests_total 16
ds4_v100_tensor_batched_tokens_total 256
```

The old auto behavior split the same 16 requests into two groups:

```text
ds4_v100_tensor_batched_groups_total 2
ds4_v100_tensor_batched_requests_total 16
ds4_v100_tensor_batched_tokens_total 256
```

## Profile Takeaways

The 16-slot profile reached `40.511669` generated tok/s under the profiler
harness. The main steady decode buckets remained F8 row-pair arena kernels,
TurboMind SM70 grouped GEMM, and grouped attention-output rows2.

Shape tracing showed the served fast path still feeds the hot F8 wrappers as
`n_tokens=1` because the event-handoff scheduler is per-slot and per-step. The
chunked scheduler exposes wider batch kernels, but it loses enough stage overlap
that end-to-end throughput regresses badly.

## Decision

Ship the 16-slot rendezvous policy and safe guarded kernel admissions. Do not
promote output-B batching, shared-down HMMA, or async slot chunking.

The next useful fusion target is not a wider batch kernel by itself. It is a
software-pipelined hot-path kernel or scheduler boundary that preserves per-slot
stage overlap while fusing packed F8/MXFP4 decode, scale application, MMA,
activation/epilogue, and writeback work. CUTLASS and TurboMind templates remain
the right reference points, but the implementation has to match the actual
served topology rather than a synthetic wide-batch shape.
