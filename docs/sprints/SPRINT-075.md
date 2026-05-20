# Sprint 075: Device-Resident Output-Head Fast Path

## Status

Complete.

## Overview

Sprint 074 showed that default-stream async HC handoff is correct but only
worth `+1.543%` at 1M/4 slots, so the next useful lever should move back into
the decode hot path.

The current greedy output-head path still behaves like a diagnostic path:

- allocate output-head temporaries on every token/slot selection;
- compute full vocab logits on gpu7;
- copy all `129280` logits back to CPU;
- scan logits on the host for top-1/top-k.

In the practical 1M/4-slot per-step profile, averaged `output_head_ms` is about
`344 ms` per response on the 16-token fixture. That is smaller than stage
decode, but it is repeated for every generated token and creates hard host/GPU
synchronization at the exact point where the async stage pipeline otherwise has
the next token ready.

Sprint 075 tested a production fast path for greedy `k=1` serving: persistent
gpu7 output-head scratch plus a CUDA top-1 reducer that returns only the
selected token/logit to the host. The existing full-logit host top-k path
remains the default because V100 evidence showed the serial device reducer made
output-head timing worse.

## Goals

1. Add a CUDA `top1` reducer for F32 logits.
2. Add persistent output-head scratch to `ds4_v100_stage_scheduler` on the
   output-head stage.
3. Route optional greedy `select_token_slot` through the device-resident top-1
   path when `k == 1` and `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`.
4. Preserve the old host full-logit top-k path for `k > 1` and as the default
   serving path.
5. Preserve selected-token correctness (`3136`) and MTP verify/commit
   correctness.
6. Measure V100 A/B evidence on the practical per-step async profile:
   - candidate fast path enabled;
   - host-logit path as the control.
7. Decide whether this should become default. Decision: no for now.

## Non-Goals

- Changing output-head dtype or quantization.
- Vocab-parallel output head across GPUs.
- Batched multi-slot output projection.
- Sampling, temperature, or non-greedy top-k serving.
- Changing MTP draft logits/top-k internals.
- Replacing the stage async pipeline or handoff mode.

## Implementation

1. Extend `ds4_gpu.h` / `ds4_cuda.cu`:
   - add `ds4_gpu_top1_f32_tensor`;
   - reduce logits on device with deterministic tie handling favoring lower
     token id;
   - copy only `{token, logit}` back to host.
2. Extend `ds4_v100_scheduler.c`:
   - add output-head scratch tensors to `ds4_v100_stage_scheduler`;
   - allocate lazily on gpu7 with enough space for HC norm, head collapse,
     output embedding/norm, and logits;
   - free scratch in scheduler close;
   - use scratch and device top-1 for `k == 1` only when
     `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`;
   - call the old allocation/full-logit path by default and for `k > 1`.
3. Keep public APIs stable:
   - no CLI flag required;
   - env opt-in only for diagnostics after the V100 A/B result.
4. Add focused validation where useful:
   - local compile for changed objects;
   - V100 selected-token smoke;
   - V100 replay/sustained smoke with candidate fast path enabled;
   - V100 same-binary host-logit control.
5. Benchmark:
   - practical profile: 1M context, 4 slots, per-step async, 16 tokens/request;
   - same binary, same fixture, candidate env on/off;
   - record generated tok/s, continuation tok/s, output-head timing, token
     matches, and GPU utilization.

## Definition of Done

- [x] Local compile passes for changed C/CUDA-facing objects where possible.
- [x] `git diff --check` passes.
- [x] V100 build passes for `tools/ds4-v100-replay` and selected-token smoke.
- [x] `cuda_v100_selected_token_smoke` passes with the candidate fast path.
- [x] `cuda_v100_selected_token_smoke` passes with the host-logit fallback.
- [x] Sustained V100 smoke returns token hex `3136` with the candidate path.
- [x] V100 A/B matrix records candidate fast path vs fallback at 1M/4 slots using
  per-step async.
- [x] Sprint report records timing deltas, output-head deltas, and default
  decision.
- [x] Vision document is updated.

## Outcome

The candidate path is correct but not a serving default.

At `ctx=1048576`, `slots=4`, `tokens=16`, `requests=4`, per-step async:

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Output-head ms | Avg GPU util |
|---|---:|---:|---:|---:|---:|
| host-logit default | `8.659254` | `8.118051` | `7389.251` | `346.461` | `19.285%` |
| device top-1 candidate | `8.697510` | `8.153916` | `7356.500` | `423.818` | `18.800%` |

Generated and continuation throughput improved by only `+0.442%`, while
output-head timing regressed by `+22.328%`. The tiny aggregate gain is below the
noise/practicality threshold and conflicts with the intended mechanism, so the
host-logit path stays default. The committed device top-1 path is opt-in with
`DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`, and
`DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1` wins if both are set.

## Decision Rule

- If the fast path is correct and improves generated tok/s or output-head
  timing materially, keep it default and keep the env fallback.
- If throughput is neutral but output-head timing drops without correctness
  risk, keep it default because it removes host synchronization and scales
  better with slots.
- If selected-token or MTP correctness regresses, or the timing evidence is not
  materially positive, disable the fast path by default and keep it diagnostic
  only.

## Risks

- Output-head projection may dominate enough that reducing host scan/readback
  gives only a small aggregate tok/s gain.
- Existing top-k diagnostics request `k > 1`; those must keep the host full
  logits path to avoid changing parity-test semantics in this sprint.
- Device top-1 must match CPU tie handling deterministically.
- Persistent scratch slightly increases gpu7 resident memory, but the expected
  footprint is under 2 MiB for scratch plus about 0.5 MiB for logits, far below
  current headroom.

## Security

No new external serving surface. The env fallback only selects an internal
output-head implementation.
