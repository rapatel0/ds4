# Sprint 123 - Production-Path Shared FFN Fusion A/B

Date: 2026-05-21

## Objective

Improve practical 16-slot/256K serving throughput without changing the
event-handoff topology that Sprint 122 validated. The first candidate was the
existing single-slot shared gate/up/SwiGLU fusion path, because the served fast
path still feeds hot F8 work as `n_tokens=1` and chunking slots to force wider
batch kernels regressed. After that A/B was too small to promote, this sprint
added a second opt-in candidate that fuses the shared-down F8 projection
epilogue with the routed/shared add.

This sprint is intentionally promotion-gated: only change production defaults if
a same-binary V100 A/B shows a correctness-preserving throughput win outside
run noise.

## Current Evidence

- Sprint 122 production-auto 16-slot/256K reaches `43.534061` generated tok/s.
- Sprint 122 shape tracing shows the hot F8 wrappers still run as
  `n_tokens=1`.
- `DS4_V100_ASYNC_SLOT_CHUNK` exposes wider kernels but loses stage overlap and
  regresses (`28.876459` at chunk 2, `18.447169` at chunk 4, `13.315378` at
  chunk 16).
- Sprint 117/120 showed the single-slot shared gate/up/SwiGLU fusion is correct
  and roughly neutral at 8 slots. It has not been re-tested under the stabilized
  16-slot production rendezvous.

## Implementation Plan

1. Sync the current worktree to the V100 build pod on `/workspace`.
2. Build the current appliance replay binary and focused F8/scheduler smokes
   with `CUDA_ARCH=sm_70 make -j80`.
3. Run a 16-slot/256K served A/B matrix:
   - production default;
   - `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1`;
   - `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1` plus
     `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=1`;
   - `DS4_V100_F8_SHARED_DOWN_ADD=1`;
   - `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1` plus
     `DS4_V100_F8_SHARED_DOWN_ADD=1`.
4. Promote only if:
   - all requests are HTTP 200;
   - token match remains `16/16`;
   - metrics report one 16-request tensor batch;
   - generated tok/s improves by at least about 2% over the same-binary default
     repeat.
5. If no candidate clears the bar, keep defaults unchanged and use the sprint
   result to narrow the next sprint to a deeper SM70 software-pipelined F8
   mainloop or TurboMind expert scheduler change.

## Files

Candidate promotion, only if A/B passes:

- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/operations/DS4-V100-APPLIANCE.md`

Always update:

- `docs/sprints/SPRINT-123.md`
- `docs/sprints/STATUS.md`
- `docs/sprints/EXPERIMENT-STATUS.md`
- `docs/sprints/VISION.md`

## Validation

Local/static:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_f8_hmma_pair_swiglu_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

Cluster correctness:

```text
tests/cuda_f8_hmma_pair_swiglu_smoke
tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43
```

Cluster serving:

```text
tools/ds4-v100-appliance-soak.sh \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --ctx 262144 --slots 16 --active-microbatch 16 \
  --tokens 16 --requests 16 --warmup-requests 1
```

## Definition Of Done

- The same-binary A/B matrix has a recorded result for the current production
  default and at least the scalar single-slot shared gate/up/SwiGLU candidate.
- Any promoted default has exact token-match correctness and a measured
  throughput win.
- If no candidate wins, the sprint records the negative evidence and leaves
  production defaults unchanged.
- Status, experiment status, and vision docs reflect the measured outcome.
- Changes are committed.

## Implementation

Added an opt-in F8 shared-down-add path:

- `ds4_gpu_arena_f8_e4m3_b128_matmul_add_f32`
- `ds4_gpu_arena_f8_e4m3_b128_matmul_batch_add_f32`
- CUDA `batch_add` row and row-pair kernels, plus a shared-down HMMA add
  epilogue variant for the existing fixed `4096 x 2048` batch shape.
- `DS4_V100_F8_SHARED_DOWN_ADD=1` launcher knob, exported to CUDA as
  `DS4_CUDA_F8_SHARED_DOWN_ADD=1`.

The layer executor uses the fused add when the per-slot path is active and, for
the batched path, when batch scratch exposes contiguous routed-output and
delta buffers. Defaults remain off pending A/B acceptance.

## Results

Static/local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build on `llamacpp-build-8gpu` under `/workspace/ds4`:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_f8_hmma_pair_swiglu_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

Correctness:

```text
cuda_f8_hmma_pair_swiglu_smoke: ok
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 tm_layers=6 token=16 pos=16 slots=16 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

Trace confirmation:

```text
ds4: f8_shape_trace kind=batch_add path=rows1 gpu=0 rows=4096 cols=2048 n_tokens=1 ... calls=96
```

Same-binary 16-slot/256K served A/B:

| Candidate | Generated tok/s | Continuation tok/s | Correctness | Decision |
|---|---:|---:|---|---|
| control | `43.070728` | `40.378807` | 16/16 token match | baseline repeat |
| `DS4_V100_F8_SHARED_DOWN_ADD=1` | `43.539555` | `40.818333` | 16/16 token match | keep opt-in |
| `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1` + `DS4_V100_F8_SHARED_DOWN_ADD=1` | `43.812630` | `41.074340` | 16/16 token match | best Sprint 123 result, still below promotion bar |

The earlier single-fusion matrix on the same Sprint 123 build produced:

| Candidate | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---|
| control | `43.613865` | `40.887999` | 16/16 token match |
| `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1` | `43.887206` | `41.144256` | 16/16 token match |
| `DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1`, rows2 | `43.627400` | `40.900688` | 16/16 token match |

## Decision

Do not promote a new default. The best Sprint 123 candidate is correct and
slightly faster than the fresh control, but the improvement is within the
observed run band and below the planned roughly 2% promotion threshold.

Keep these diagnostics opt-in:

```text
DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1
DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=1
DS4_V100_F8_SHARED_DOWN_ADD=1
```

The result reinforces the current direction: small launch-count fusions are not
enough. The next production sprint should target a larger execution boundary,
most likely TurboMind route-row compaction/persistent grouped expert execution
or a CUTLASS/TurboMind-inspired software-pipelined FFN kernel that fuses
decode, staging, HMMA, SwiGLU, and epilogue work without sacrificing the
current per-step stage overlap.
