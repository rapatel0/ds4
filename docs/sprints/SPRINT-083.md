# Sprint 083: Opt-In TurboMind Runtime Routed FFN Bridge

## Status

Complete.

## Overview

Sprint 082 proved the copied TurboMind MXFP4 grouped GEMM at the adapter-smoke
boundary. Sprint 083 wires that path into the DS4 CUDA runtime behind an
explicit opt-in flag while preserving the existing source-MXFP4 arena kernels
as the default and fallback.

The implementation is intentionally transient: it packs one expert matrix at a
time, runs the grouped TurboMind GEMM, then frees the packed buffers. This
avoids overfilling 32 GB V100s while proving runtime semantics. It is not the
final performance layout. The production path should move TurboMind-ready
expert packs into the offline shard format so runtime does not duplicate or
repack the largest tensors.

## Goals

1. Add a runtime `dlopen` bridge for copied `libggml-turbomind.so`.
2. Add an opt-in routed FFN path behind `DS4_V100_TURBOMIND_ROUTED_FFN=1`.
3. Build expert-grouped offsets and activation rows from device-resident
   `selected_i32` and `weights_f32`.
4. Run TurboMind grouped gate, up, and down GEMMs through the existing DS4
   arena wrapper boundary.
5. Preserve fallback to the source-MXFP4 arena path unless
   `DS4_V100_TURBOMIND_STRICT=1`.
6. Validate the runtime wrapper on V100 against the existing arena reference.

## Non-Goals

- Making TurboMind the production default.
- Persistently caching TurboMind packs for every layer.
- Replacing the offline V100 pack format.
- Sustained tok/s benchmarking of transient repack mode as a production result.

## Definition of Done

- [x] Runtime compiles with the copied TurboMind C ABI header.
- [x] Replay binary links with the new dynamic loading dependency.
- [x] Adapter smoke validates the direct TurboMind adapter.
- [x] Adapter smoke validates the DS4 runtime wrapper path.
- [x] Deployment env documents the opt-in flags.
- [x] V100 log is recorded under `logs/from-cluster/`.
- [x] Artifacts are committed.

## Result

`SHIP_RUNTIME_BRIDGE`.

Cluster result:

```text
cuda_v100_turbomind_adapter_smoke: experts=8 routes=6 gate_kpack=0x341321 down_kpack=0x341321 max_abs=0.00129318 rel=0.000258549 bad=0
cuda_v100_turbomind_adapter_smoke: runtime_wrapper max_abs=0.00129318 rel=0.000258549 bad=0 host_ms=43.298
cuda_v100_turbomind_adapter_smoke: PASS
```

Decision: keep `DS4_V100_TURBOMIND_ROUTED_FFN=0` by default. The next kernel
sprint should stop transient runtime packing and create an offline
TurboMind-expert pack layout, or a bounded per-stage cache admitted by the
memory planner, before treating TurboMind as a throughput path.
