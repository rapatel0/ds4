---
sprint: 019
title: V100 Integrated Single-Layer Runtime Slice
status: intent
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../../architecture/DS4-V100-LAYOUT.md
---

# SPRINT-019 Intent: V100 Integrated Single-Layer Runtime Slice

## Seed Prompt

Continue toward the DS4 V100 appliance goal, but use larger implementation
sprints with parallelizable workstreams. Stop planning tiny primitives in
isolation; ship a real runtime slice and validate it on V100 hardware. CPU
validation is useful only as a reference and regression guard, not as deployment
evidence.

## Orientation Summary

- Sprint 018 shipped descriptor-bound attention projection, residual, and norm
  work from real layer-2 source bytes through `ds4_v100_layer_state`, but it
  explicitly did not prove full attention semantics.
- The current gate passes on the V100 pod and reports `ready=false` because the
  appliance still lacks a coherent scheduler-owned layer output, real-model
  selected-token decode, public serving, MTP, and throughput benchmarking.
- `docs/architecture/DS4-V100-LAYOUT.md` is the architecture anchor. Sprint 019
  should not re-litigate sharding; it should use the layer-sharded 8-GPU
  baseline and the source-FP8/MXFP4/F16-KV starting layout.
- The actionable Sprint 018 follow-ups are full attention softmax over
  raw/compressed KV, combined attention plus FFN layer output, and production
  arena reuse.
- Direct hardware validation should use one or two visible V100s for iteration,
  then the full appliance gate once the slice is stable.

## Relevant Codebase Areas

- `ds4_v100_layer_state.h`, `ds4_v100_layer_state.c`: scheduler-owned
  descriptor/state surface.
- `ds4_gpu.h`, `ds4_cuda.cu`: CUDA source-format projection, compressor,
  attention, RMSNorm, router, and FFN kernels.
- `tests/cuda_v100_descriptor_bound_attention_smoke.c`: real-byte attention
  projection/residual/norm smoke.
- `tests/cuda_v100_descriptor_bound_ffn_smoke.c`: real router-selected
  descriptor-bound FFN smoke.
- `tests/cuda_v100_projection_attention_smoke.c`: synthetic attention and
  compressor references useful for raw/compressed KV semantics.
- `tools/ds4-v100-gate.sh`: appliance readiness gate.
- `docs/architecture/DS4-V100-LAYOUT.md`: sharding, memory, kernel, dtype, and
  tensor-parallel planning anchor.

## Sprint Goal

Build a callable, scheduler-owned, descriptor-bound V100 layer execution slice
for one representative ratio-4 layer. The slice should produce a coherent next
hidden state by combining:

1. real source-byte attention projection inputs;
2. raw SWA plus compressed KV visibility;
3. attention sink/score handling and softmax over the visible rows;
4. attention output projection, residual, and FFN pre-norm;
5. real router-selected FFN execution; and
6. final residual next-hidden output.

The sprint should leave behind a reusable execution surface, not only a test
composition.

## Parallel Workstreams

These lanes are intentionally separable so subagents or shell sessions can work
in parallel without touching the same write set.

| Lane | Owner Scope | Files | Output |
|---|---|---|---|
| A: attention semantics | raw/compressed KV reference and GPU call sequence | `ds4_cuda.cu`, attention tests | bounded softmax output over raw plus compressed rows |
| B: layer executor API | callable layer-slice surface using layer state | `ds4_v100_layer_execute.*`, `ds4_v100_layer_state.*` | one function that executes layer-2 slice into next hidden |
| C: integrated smoke/gate | test bench and readiness gate wiring | `tests/cuda_v100_integrated_layer_smoke.c`, `Makefile`, `tools/ds4-v100-gate.sh` | V100 runnable integrated layer check |
| D: hardware validation | direct-host run scripts/log capture | docs/logs only unless a helper is needed | one-GPU iteration logs, full gate log |

## Constraints

- Preserve the source model's intelligence path: source FP8 dense tensors and
  MXFP4 routed experts remain the starting point. INT8 remains an option, not
  the default, unless correctness and memory evidence justify it.
- V100 has no native BF16, FP8, or FP4 tensor-core execution. Runtime compute
  must use explicit source decode/unpack into FP16 or existing low-bit/integer
  kernels with FP32 reductions where needed.
- Do not materialize persistent dequantized weights in VRAM. Use kernel-local
  or bounded scratch conversion.
- Use F16 KV first. F8 KV is deferred until the semantic layer output works.
- Use the baseline layer-sharded topology from
  `docs/architecture/DS4-V100-LAYOUT.md`; tensor-parallel variants remain
  analysis candidates, not Sprint 019 implementation scope.
- Validation must run on actual V100 hardware before the sprint can ship.

## Success Criteria

- A reusable layer execution surface exists and is used by the integrated smoke.
- The integrated smoke consumes real pack-index descriptors and real source
  bytes for a ratio-4 layer.
- The smoke validates attention softmax over raw plus compressed rows, then
  composes attention output, residual, FFN pre-norm, router-selected FFN, and
  next-hidden residual.
- CPU/source-format references exist for the bounded check and are kept as
  regression tests.
- V100 hardware evidence is captured from direct one-GPU iteration and the full
  appliance gate.
- The readiness gate remains honest: it may still report `ready=false`, but the
  missing reason list should shrink if full layer scheduling is genuinely
  covered.

## Verification Strategy

- Local build checks:
  - `make tests/v100_layer_state_smoke`
  - `make tests/cuda_v100_descriptor_bound_attention_smoke.o`
  - `make tests/cuda_v100_descriptor_bound_ffn_smoke.o`
  - `make tests/cuda_v100_integrated_layer_smoke.o`
- Direct V100 iteration:
  - `ssh ubuntu@192.168.102.5`
  - run with `CUDA_VISIBLE_DEVICES=0` for single-GPU layer-2 iteration;
  - use `CUDA_VISIBLE_DEVICES=0,1` only if boundary or staging behavior is
    needed before the full gate.
- Final hardware gate:
  - run `tools/ds4-v100-gate.sh` with the source GGUF and pack index;
  - capture logs under `docs/sprints/drafts/SPRINT-019-*`.

## Uncertainty Assessment

| Area | Level | Notes |
|---|---|---|
| Correctness | High | The exact DS4 attention composition, sink handling, compressed-row selection, and output projection path are the current blocker. |
| Scope | Medium | The sprint is larger than recent ones, but remains bounded to one representative layer and one execution surface. |
| Architecture | Medium | Layer-sharded topology is stable; the uncertain part is how much of the semantic attention path can reuse existing kernels without changing data layout. |
| Hardware | Medium | Earlier smokes passed on V100, but the integrated slice will stress larger scratch and more kernel transitions. |

## Open Questions

1. Does the existing attention kernel already model the DS4 ratio-4 indexer
   visibility exactly enough for a real layer-2 bounded reference?
2. Should Sprint 019's layer executor expose only single-token decode first, or
   also a tiny prefill microbatch path?
3. Can the gate remove `full_layer_scheduler` after a one-layer executor ships,
   or should that reason remain until all 43 layers can be walked?
4. Which minimal direct-host artifact is worth keeping: a checked-in helper
   script or only logged commands?

