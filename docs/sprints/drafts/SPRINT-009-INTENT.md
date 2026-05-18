# SPRINT-009 Intent: V100 Prefill And Compressed KV Execution

## Seed Prompt

Continue the DS4 V100 appliance vision after Sprint 008. Implement the first
V100 source-layout prompt prefill and compressed KV/indexer execution surface
without unlocking normal source-layout serving.

## Orientation Summary

- Sprint 008 shipped the correctness and safety gates needed for runtime work:
  `tools/ds4-source-oracle-vector`, source-layout guard checks, exact F16 KV
  admission by layer/stage/context/slot, source dtype hardening, and a bounded
  CUDA F8_E4M3_B128 row-decode anchor.
- `docs/architecture/DS4-V100-LAYOUT.md` remains the topology contract: 8-GPU
  layer ownership, pure device-resident weights, F16 KV first, no persistent
  dequantized large source tensors, FP16 HMMA with FP32 accumulation for dense
  math, and F32 only for small control/reduction/debug paths.
- Current CUDA surfaces already include diagnostic arena residency, BF16 row
  gather, F8 row decode, HC relay, indexer, attention prefill, raw KV store,
  and compressor kernels. They are not yet assembled into a source-layout V100
  prefill runtime.
- Sprint 009 should consume Sprint 008's derived KV budget rather than
  reintroducing coarse KV estimates.
- Normal source-layout generation must remain guarded. Any new runtime path
  must be diagnostic/test-only until V100 correctness is demonstrated.

## Relevant Codebase Areas

- `ds4_v100_context.[ch]`: layer/stage map, descriptor policy, KV admission,
  memory reserve, skeleton report.
- `ds4_v100_context_cuda.cu`: CUDA resource wrapper around the V100 context.
- `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_gpu_arena_stub.c`: diagnostic arena and CUDA
  primitive surfaces.
- `ds4_source_formats.[ch]`: CPU reference source-format helpers.
- `tools/ds4-source-oracle-vector.c`: source oracle and guard runner.
- `tools/ds4-v100-context-smoke.c`: executable V100 context/KV plan report.
- `tests/cuda_source_dtypes_smoke.c`, `tests/cuda_v100_context_smoke.c`,
  `tests/cuda_hc_relay_smoke.c`: V100 validation patterns.

## Success Criteria

- A V100 prefill/KV diagnostic path allocates F16 KV arenas from the derived
  stage budget for explicit `ctx` and `slots`.
- At least one ratio-4 layer and one ratio-128 layer can execute a bounded
  device-side prefill KV update over synthetic or pack-backed source-layout
  inputs.
- F8 source tensors used by the path consume the source packed layout through
  the Sprint 008 row-decode pattern or a bounded no-persistent-dequant tile.
- Raw SWA, compressed attention KV, and ratio-4 indexer KV/state updates are
  visible in a stable report or smoke output.
- Device results are compared against CPU/source helper references for the
  bounded slice.
- `tools/ds4-source-oracle-vector --guards-only` still passes on the real source
  model; normal source-layout generation remains fail-closed.
- CUDA validation passes on V100 `sm_70`.

## Verification Strategy

- Local model-less build/tests: source dtype smoke, V100 context smoke, and any
  new prefill/KV model-less smoke.
- Cluster V100 validation:
  - `tools/ds4-source-oracle-vector --guards-only` against
    `/models/DSv4-Flash-256e-fixed.gguf`.
  - `tests/cuda_v100_context_smoke --production --kv-ctx ... --kv-slots ...`.
  - New CUDA prefill/KV smoke built with `CUDA_ARCH=sm_70`.
- Archive all logs under `docs/sprints/drafts/`.
- Report `SHIP`, `EXTEND`, or `STOP` with exact unsupported paths and guards.

## Uncertainty Assessment

- Correctness: High. The prefill/KV path touches DS4 compressed attention,
  indexer state, source formats, and stage memory ownership.
- Scope: Medium-High. Full 43-layer source-layout prefill may be too large;
  the sprint should explicitly ship a bounded runtime slice rather than
  silently expanding into full decode.
- Architecture: Medium. The layer-owned topology is stable, but the right
  boundary between diagnostic prefill harness and future production scheduler
  needs care.

## Open Questions

1. Should Sprint 009's `SHIP` gate require pack-backed tensors from the real
   source shard, or is synthetic source-layout input enough for the first KV
   execution slice?
2. Which layer pair should anchor the smoke: layer 2 for ratio-4/indexer and
   layer 3 for ratio-128, or a later pair owned by a nonzero GPU to exercise
   stage-local allocation away from gpu0?
3. Should the first dense source-format projection use F8 decode-to-F32
   diagnostic output, F8 decode-to-F16 scratch, or a direct F8-to-F16 HMMA tile?
4. What tolerance should be used for CUDA-vs-CPU reference comparisons in the
   first F16 KV update path?

## Vision Context

Sprint 009 is the runtime bridge between Sprint 008's executable contracts and
Sprint 010 deployment. It should make prefill/KV execution real enough that the
next sprint can package a guarded CLI/server path, but it should not attempt
throughput optimization, MTP, broad tensor parallelism, or normal source-layout
serving unlocks.
