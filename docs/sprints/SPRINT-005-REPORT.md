---
sprint: 005
title: First Resident BF16 Gather/Expand Probe
date: 2026-05-17
status: shipped
---

# SPRINT-005 Report

## Verdict

`SHIP`.

Sprint 005 proves that source BF16 bytes resident in a `ds4_gpu_arena` can be
addressed by descriptor, read by a CUDA kernel on the owning V100, expanded to
F32, and verified bit-exactly against the source bytes. This is deliberately
not a production BF16 performance path: V100 has no native BF16 tensor-core
execution. The shipped probe is a residency/addressing/dtype diagnostic that
keeps the source-model generation guard active.

## What Shipped

- Added `ds4_gpu_bf16_matrix_view` and
  `ds4_gpu_arena_bf16_row_gather_f32`.
- Implemented matching host-stub and CUDA arena row-gather paths.
- Added `tests/bf16_probe_smoke.c` for model-less BF16 bit-pattern coverage.
- Added `tests/cuda_bf16_probe.c` for direct V100 CUDA validation.
- Extended `tools/ds4-v100-residency-smoke` with `--bf16-probe`,
  `--probe-row`, `--probe-samples`, and `--probe-only`.
- Extended the synthetic residency smoke to exercise both GGUF and shard
  provider probe-only paths.

## Validation

Local laptop validation:

```bash
make cpu tests/bf16_probe_smoke tests/gpu_arena_smoke \
  tests/pack_index_smoke tools/ds4-v100-residency-smoke
./tests/pack_index_smoke
./tests/gpu_arena_smoke
./tests/bf16_probe_smoke
tests/residency_smoke_synthetic.sh
git diff --check
```

Cluster validation on `llamacpp-build-8gpu`, `CUDA_ARCH=sm_70`:

```bash
make clean
make cpu tests/bf16_probe_smoke tests/gpu_arena_smoke \
  tests/pack_index_smoke tools/ds4-v100-residency-smoke \
  tests/cuda_bf16_probe CUDA_ARCH=sm_70
./tests/pack_index_smoke
./tests/gpu_arena_smoke
./tests/bf16_probe_smoke
./tests/cuda_bf16_probe
tests/residency_smoke_synthetic.sh
```

All passed. The direct CUDA probe printed `cuda_bf16_probe: ok`.

Durable cluster log:

- `docs/sprints/drafts/SPRINT-005-CUDA-SYNTHETIC.log`

Real model probes:

- `docs/sprints/drafts/SPRINT-005-BF16-PROBE-GGUF.log`
- `docs/sprints/drafts/SPRINT-005-BF16-PROBE-SHARD.log`

Both real probes used `token_embd.weight`, shape `[4096x129280]`, byte length
`1059061760`, owning GPU `0`, and rows `0`, `1`, and `12345`. Both logged 48
expected/actual F32 bit samples and ended with `bf16_probe_result	OK` and
`result	OK`.

Observed GPU 0 facts from both provider paths:

- arena bytes: `1059061760`
- used bytes: `1059061760`
- memory kind: `device`
- free after upload: `32686735360`
- total memory: `34072559616`

Guard validation:

- `docs/sprints/drafts/SPRINT-005-GUARD.log`

Normal source-model generation still fails closed with:

```text
native DS4-Flash source layout is recognized, but V100 FP8/MXFP4 execution kernels are not wired into runtime yet
```

## Deviations

The sprint title and vision language were tightened from "BF16 compute" to
"BF16 gather/expand". That correction matters: V100 cannot run native BF16
tensor-core math, so production execution should choose V100-native FP16
tensor-core paths or the existing low-bit/integer kernel families after offline
packing, not treat BF16 as a fast compute format.

The shard-provider real probe used a minimal `gpu0.weights` slice containing
only `token_embd.weight` because the disposable cluster pod did not have the
full persistent shard directory mounted. The full-residency shard provider was
already proven in Sprint 004; Sprint 005 only needed the resident
gather/expand diagnostic over the probed tensor span.

## Next

Sprint 006 should define the production execution context and make the V100
execution-format policy explicit:

- BF16 source tensors: expand to F32 only for small control/verification paths;
  otherwise convert to FP16 where tensor-core math is required.
- FP8/MXFP4 expert and attention tensors: use offline V100 packing plus
  kernel-specific unpack/dequant into FP16 or integer fragments.
- Avoid repeated per-token BF16/FP8/FP4 to F32 staging on large GEMMs.
