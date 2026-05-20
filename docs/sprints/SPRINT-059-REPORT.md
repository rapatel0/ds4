# Sprint 059 Report: Persistent Layer Batch Scratch

## Result

`SHIP`.

## Changes Implemented

1. Added reusable layer batch scratch.
   - `ds4_v100_layer_batch_scratch` owns HC batch temporaries and FFN batch
     temporaries sized for `DS4_V100_LAYER_MAX_BATCH`.
   - Scratch is lazily allocated on the stage GPU the first time a multi-slot
     layer batch executes.
   - Direct callers can still pass `NULL` and use the old allocation/free path.
2. Wired one scratch object per stage scheduler.
   - `ds4_v100_stage_scheduler` owns and frees the scratch object.
   - `ds4_v100_stage_scheduler_decode_hc_batch` passes scratch into layer
     configs only for multi-slot layer batches.
3. Enabled multi-slot layer batching by default after benchmark evidence.
   - `DS4_V100_BATCH_LAYER_FFN=1` is no longer required.
   - `DS4_V100_BATCH_LAYER_FFN=0`, `off`, or `false` disables the path.

## Validation

Local:

```bash
cc -fsyntax-only -I. ds4_v100_layer_execute.c
cc -fsyntax-only -I. ds4_v100_scheduler.c
make ds4_v100_layer_execute.o ds4_v100_scheduler.o tools/ds4-v100-replay.o
git diff --check
```

Cluster build:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-replay \
  tests/cuda_v100_integrated_layer_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_mxfp4_moe_smoke
```

Focused V100 smokes:

```text
cuda_v100_mxfp4_moe_smoke: ok
cuda_v100_integrated_layer_smoke: layer=2 ... ok
cuda_v100_stage_scheduler_smoke --slots 2: ok
DS4_V100_BATCH_LAYER_FFN=1 cuda_v100_stage_scheduler_smoke --slots 2: ok
DS4_V100_BATCH_LAYER_FFN=1 cuda_v100_full_scheduler_smoke --slots 2: ok
cuda_v100_stage_scheduler_smoke --slots 2 after default flip: ok
DS4_V100_BATCH_LAYER_FFN=0 cuda_v100_stage_scheduler_smoke --slots 2: ok
```

Real replay correctness: first token id `926`, text `16`, hex `3136`.

## Sustained Decode Comparison

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | max GPU util |
|---|---:|---:|---:|---:|---:|
| Sprint 058 default | 1 | 3.583987 | 3.359988 | 11.141% | 20.000% |
| Sprint 059 pre-flip default | 1 | 3.584540 | 3.360506 | 10.944% | 20.000% |
| Sprint 059 default-batched | 1 | 3.587298 | 3.363092 | 11.134% | 20.000% |
| Sprint 058 default | 2 | 3.704572 | 3.473036 | 11.162% | 20.000% |
| Sprint 059 pre-flip default | 2 | 3.697740 | 3.466631 | 11.428% | 20.000% |
| Sprint 059 scratch batch | 2 | 3.855284 | 3.614329 | 11.020% | 20.000% |
| Sprint 059 default-batched | 2 | 3.862932 | 3.621499 | 11.434% | 20.000% |

The final default-batched path improves two-slot generated tok/s by about
`4.27%` over Sprint 058 and about `4.46%` over the Sprint 059 pre-flip default
run. The one-slot path is effectively unchanged, as expected.

Artifacts:

- `logs/from-cluster/sprint059-default/sustained_decode.tsv`
- `logs/from-cluster/sprint059-ffn-scratch/sustained_decode.tsv`
- `logs/from-cluster/sprint059-default-batched/replay.json`
- `logs/from-cluster/sprint059-default-batched/sustained_decode.tsv`
- matching per-case `result.json`, status snapshots, `server.log`, and
  `gpu_util.csv`

## Assessment

Persistent scratch fixed the reason the batched layer path was previously
disabled. The path is now correct on focused and full 8-stage smokes, improves
two-slot sustained throughput, and is enabled by default for multi-slot decode.

This is still an incremental runtime-shape gain, not a breakthrough. Average
GPU utilization remains near `11%`. The next optimization needs to remove the
remaining per-slot FFN input copy or change the routed MXFP4 kernel to consume
slot input pointers directly.
