# Sprint 081: Copy And Prove TurboMind MXFP4 Grouped GEMM

## Status

Complete. Outcome: `SHIP_PROOF`.

## Overview

Sprint 080 proved the copied-source workflow with tc-grid INT8, but the result
also confirmed the limits of that path for DS4 routed decode. TurboMind is a
better next candidate because the prior work shows stronger V100 tensor-core
utilization and its MXFP4 deinterleave matches DS4's source expert layout.

This sprint copies the TurboMind C ABI wrapper and required lmdeploy
`turbomind` support source into `ds4`, adapts the build so it no longer points
at `~/repos/deepseek`, and proves the grouped MXFP4 GEMM path on V100 from the
copied source.

## Goals

1. Copy the relevant TurboMind wrapper and lmdeploy `turbomind` source into
   this repository.
2. Adapt the copied CMake so the default `LMDEPLOY_SRC` points at the copied
   tree.
3. Build `libggml-turbomind.so` from inside `ds4` on V100.
4. Run the grouped MXFP4 compare smoke from copied source:
   - DS4 down shape `N=4096,K=2048`;
   - DS4 gate/up shape `N=2048,K=4096`;
   - decode and prompt-like token counts.
5. Record whether TurboMind is ready for a DS4 hot-path adapter sprint.

## Non-Goals

- Wiring TurboMind into the DS4 scheduler hot path.
- Replacing the existing source-MXFP4 scalar kernels.
- Making TurboMind a production default.
- Solving continuous batching or route sorting.
- Copying unrelated lmdeploy model, serving, or Python code beyond what the
  grouped GEMM build needs.

## Implementation

1. Add copied source under `kernels/turbomind/`:
   - `ggml-turbomind/` from `ggml/vendor/turbomind`;
   - `lmdeploy/src/turbomind/` support tree;
   - `lmdeploy/LICENSE`.
2. Patch copied `ggml-turbomind/CMakeLists.txt`:
   - default `LMDEPLOY_SRC` to `../lmdeploy/src`;
   - keep CUDA architecture explicit through the build command;
   - keep tests enabled for grouped compare.
3. Add a repository-local README with source paths and Sprint 081 intent.
4. Validate on the V100 pod:
   - configure with CMake from `kernels/turbomind/ggml-turbomind`;
   - build `ggml-turbomind` and `test_ggml_turbomind_grouped_compare`;
   - run grouped compare against the copied shared library.

## Definition of Done

- [x] TurboMind/lmdeploy source needed for grouped GEMM lives in this repo.
- [x] Build does not reference `~/repos/deepseek`.
- [x] `libggml-turbomind.so` builds on V100 from copied source.
- [x] Grouped MXFP4 compare passes on V100 for DS4 gate/up and down shapes.
- [x] Report records build command, smoke output, and any dependency caveats.
- [x] Vision and architecture docs are updated with the decision.
- [x] Artifacts are committed.

## Decision Rule

- If copied TurboMind does not build, stop and record the exact missing
  dependency or source surface.
- If it builds but grouped MXFP4 compare fails, do not plan hot-path integration
  until the packing/layout bug is localized.
- If it builds and compare passes, plan the next sprint around a DS4 routed
  expert adapter: pack source MXFP4 experts at load, route/gather into grouped
  offsets, run gate/up grouped GEMMs, apply SwiGLU and route weights, run down
  grouped GEMM, scatter/sum back into HC.

## Risks

- The copied build may still need external header-only dependencies such as
  CUTLASS/fmt, even though the TurboMind kernel source is local.
- Build time may be substantial on the V100 pod.
- Grouped GEMM correctness does not by itself prove end-to-end scheduler
  speedup; route sorting/gather/scatter costs still matter.

## Security

No new serving surface. This sprint adds copied kernel/library source and a
local V100 test binary only.
