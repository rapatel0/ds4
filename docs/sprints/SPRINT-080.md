# Sprint 080: Copy V100 Low-Bit Kernel Sources Into ds4

## Status

Complete. Outcome: `SHIP_PROOF_ONLY`.

## Overview

Sprint 079 proved that another scalar MXFP4 row-level reshaping is not enough:
the appliance still spends most of decode time in low-utilization routed expert
kernels. The next step is to stop treating the prior `~/repos/deepseek`
tc-grid/TurboMind work as only design evidence. Any kernel path we intend to
use must have its source copied into this repository, adapted here, and proven
from this repository on V100.

This sprint starts with the smallest useful copied-source proof: tc-grid's V100
INT8 `v13_rf_v6` HMMA kernel. That path is not presumed to be the final routed
expert backend. TurboMind remains the likely preferred MXFP4 grouped-GEMM path
if its larger lmdeploy/CUTLASS support tree can be copied and built cleanly
inside `ds4`.

## Goals

1. Copy the relevant tc-grid V100 source files from `~/repos/deepseek` into
   `ds4`.
2. Add a `ds4` build target that compiles the copied tc-grid source directly
   for `sm_70`.
3. Add a V100 smoke/bench that launches the copied `v13_rf_v6` INT8 HMMA kernel
   from this repository.
4. Validate correctness on a bounded matrix and collect a DS4-shaped timing
   sample.
5. Record the TurboMind decision point: it is probably the better final MXFP4
   path, but it needs a larger copied source surface and a grouped-GEMM smoke.

## Non-Goals

- Making INT8 the model default.
- Repacking DS4 MXFP4 routed experts into INT8 in the runtime.
- Wiring tc-grid into the hot-path scheduler before a copied-source smoke is
  proven.
- Copying the full TurboMind/lmdeploy tree in this sprint unless the tc-grid
  proof blocks immediately.
- Changing public serving behavior.

## Implementation

1. Add copied source under `kernels/tc-grid/`:
   - `include/tc_grid.h`;
   - `include/dispatch.h`;
   - `kernels/mma_sm70.cuh`;
   - `kernels/v12_kernels.cuh`;
   - `kernels/v13_kernels.cuh`;
   - `LICENSE.turbomind`.
2. Add `tests/cuda_v100_tc_grid_int8_smoke.cu`:
   - build deterministic FP32 activations, INT8 weights, and FP16 scales;
   - launch copied `mm_int8_lut_v13_rf_v6<128,128,16,16>`;
   - compare a small case to a CPU FP32 reference;
   - time a DS4-like shape, defaulting to `M=128, N=2048, K=4096`.
3. Extend `Makefile` with a copied-source tc-grid CUDA target.
4. Run V100 validation:
   - `CUDA_ARCH=sm_70 make tests/cuda_v100_tc_grid_int8_smoke`;
   - `./tests/cuda_v100_tc_grid_int8_smoke`;
   - at least one larger timing run, for example `128 2048 4096 100`.

## Definition of Done

- [x] Copied tc-grid source lives in this repository.
- [x] The new tc-grid smoke builds from `ds4` without referencing
  `~/repos/deepseek`.
- [x] Bounded correctness check passes on V100.
- [x] DS4-shaped tc-grid timing is recorded.
- [x] Sprint report states whether tc-grid is only a proof point or a candidate
  for hot-path integration.
- [x] Vision document is updated with the kernel-copy decision.
- [x] Artifacts are committed.

## Decision Rule

- If copied tc-grid does not build or pass correctness, do not integrate it into
  the model path.
- If copied tc-grid works but is an INT8-only detour, keep it as an integration
  and benchmarking proof while prioritizing copied TurboMind MXFP4 grouped GEMM.
- If copied tc-grid shows strong V100 throughput and the memory planner can
  admit an INT8 expert pack without overfilling VRAM, plan a separate quality
  gate before any runtime use.

## TurboMind Follow-Up

TurboMind has stronger evidence for tensor-core utilization and native DS4
MXFP4 expert compatibility, but the source surface is larger. The next sprint
should copy and build a minimal TurboMind grouped-GEMM probe if Sprint 080
confirms that copied-source kernel development can run cleanly from `ds4`.

Expected copied surface:

- `ggml/vendor/turbomind/include/ggml-turbomind-api.h`;
- `ggml/vendor/turbomind/api.cc`;
- `ggml/vendor/turbomind/ggml-turbomind-deinterleave.cu`;
- `ggml/vendor/turbomind/ggml-turbomind-deinterleave.h`;
- required `research/lmdeploy/src/turbomind/` support directories;
- local CUTLASS/fmt/concurrentqueue handling.

The target smoke should pack one DS4-style MXFP4 expert matrix, run grouped
gate/up and down GEMMs with FP16 activations, and compare against the existing
source-MXFP4 reference.

## Risks

- tc-grid INT8 may be a useful proof but not the right model-quality path.
- TurboMind may require more build-system work than a single sprint can absorb.
- A copied standalone GEMM can look good while still requiring substantial
  gather/scatter, SwiGLU, route weighting, and down-sum work in the scheduler.

## Security

No new serving surface. This sprint adds copied CUDA source and a local V100
test binary only.
