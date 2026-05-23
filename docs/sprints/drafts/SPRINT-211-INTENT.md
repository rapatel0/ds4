# Sprint 211 Intent - TP8 TurboMind MXFP4 Expert Body

Date: 2026-05-23

## Seed Prompt

Continue the high-throughput practical-serving vision after Sprint 210. The TP
path remains completely separate from the PP/layer scheduler. Do not add a
generic scheduler and do not retrofit TP into `ds4_v100_scheduler.*`.

## Orientation Summary

- Sprint 209 proved the all-8-GPU TP boundary and sharded KV allocation inside
  a separate TP-only executable.
- Sprint 210 replaced synthetic compute with resident FP16 Tensor Core GEMM
  fixture work and showed useful work can live inside the TP8 boundary.
- The remaining gap is precision/layout fidelity: DS4 experts are low-bit
  MXFP4/FP8, and the appliance already has TurboMind MXFP4 kernels that pack,
  unpack, and compute on V100.
- Existing TurboMind TP split tests prove 2-GPU middle-dimension splitting, but
  not all-8 TP8 execution at the 32-slot target.
- Current worktree still has unrelated dirty Sprint 207 runtime/kernel files;
  do not stage or clean them up.

## Relevant Code Areas

- `tools/ds4-v100-tp8-real-layer-smoke.cu`: Sprint 210 TP8 resident layer
  fixture and recursive-doubling reduction pattern.
- `kernels/turbomind/ggml-turbomind/test_tp_split_2gpu.cpp`: existing
  TurboMind MXFP4 middle-dimension split correctness and benchmark reference.
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`: public ABI.
- `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp`: fixture
  packing helpers and fixed-shape probe usage.
- `Makefile`: add TP-only CUDA tool target.

## Constraints

- New TP-only file(s) only.
- No PP scheduler changes.
- No launcher default changes.
- No model weights in logs; use deterministic synthetic MXFP4 fixtures.
- Use public TurboMind C ABI through `dlopen`, not direct coupling to internal
  TurboMind C++ symbols.
- Validate on all eight V100s.

## Success Criteria

- Build a new TP-only executable that:
  - packs deterministic MXFP4 gate/up and down fixtures;
  - creates a full single-GPU reference expert FFN;
  - creates eight middle-shard TP participants;
  - runs TurboMind gated-SiLU and down GEMMs on all participants;
  - reduces/sums the eight partial down outputs;
  - compares TP8 sum against the full reference;
  - reports full reference time, TP8 compute time, reduce/copy time, and
    speedup.
- Run route shapes matching the practical target:
  - `tokens_per_active=16` / `routes=96`;
  - `tokens_per_active=32` / `routes=192`;
  - optional `tokens_per_active=64` / `routes=384` if stable.
- Preserve the 32-slot / 128K-256K planning direction in docs, but keep this
  sprint focused on routed-FFN compute.

## Verification Strategy

- Local hygiene: `git diff --check`; macOS CUDA target prints CUDA-required.
- V100 build:
  `make -j80 tools/ds4-v100-tp8-turbomind-ffn-smoke CUDA_ARCH=sm_70`.
- Build or locate `libggml-turbomind.so` in the V100 workspace.
- V100 runs at required route shapes.
- Logs copied to `logs/from-cluster/sprint211-tp8-turbomind-ffn/`.
- Update `docs/sprints/SPRINT-211.md`, `docs/sprints/STATUS.md`, and
  `docs/sprints/VISION.md`.

## Uncertainty

- Correctness: Medium. TP8 sums eight low-bit partials and compares against a
  full low-bit path; tolerances must be explicit.
- Scope: Medium. If generic TurboMind kernels reject `mid_shard=256`, fallback
  should test `TP4` or `mid_shard=512` as a documented blocker rather than
  modifying the scheduler.
- Architecture: Low. Separate TP-only executable is the agreed shape.

## Open Questions

- Does TurboMind's generic MXFP4 path support the smaller TP8 shard dimensions
  cleanly on V100?
- Is TP8 compute speedup still visible after the eight-way output reduction?
- Is the next sprint low-bit TP8 runtime ownership or sharded attention/KV?

## Vision Context

Sprint 211 is the precision/layout fidelity gate after Sprint 210. If it
passes, TP8 has cleared topology, real Tensor Core fixture, and real low-bit
expert body gates. Serving integration is still future work and must remain in
new TP-only runtime files.
