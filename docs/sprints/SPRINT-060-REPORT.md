# Sprint 060 Report: Pointer-Input Routed FFN Batch

## Result

`SHIP`.

## Changes Implemented

1. Added a pointer-input grouped MXFP4 routed FFN primitive.
   - New public API:
     `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32`.
   - The grouped gate/up/SwiGLU kernel now reads per-token input rows through a
     device pointer table instead of a contiguous `[token x hidden]` tensor.
   - The pointer table is a caller-provided GPU tensor, so it is device-local
     and safe across the 8-stage multi-GPU scheduler.
2. Removed `input_batch_t` from the V100 layer batch path.
   - `execute_ffn_delta_batch` now passes the per-slot `ffn_inputs[]` directly
     to the routed MXFP4 batch primitive.
   - Persistent layer scratch now owns `ffn_input_ptrs` instead of
     `ffn_input_batch`.
3. Extended focused MXFP4 smoke coverage.
   - `tests/cuda_v100_mxfp4_moe_smoke.c` now validates the pointer-input batch
     API using separate per-slot input tensors.

## Validation

Local:

```bash
cc -fsyntax-only -I. ds4_v100_layer_execute.c
cc -fsyntax-only -I. tests/cuda_v100_mxfp4_moe_smoke.c
make ds4_v100_layer_execute.o tests/cuda_v100_mxfp4_moe_smoke.o
git diff --check
```

Cluster build:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-replay \
  tests/cuda_v100_mxfp4_moe_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke
```

Focused V100 smokes:

```text
cuda_v100_mxfp4_moe_smoke: ok
cuda_v100_stage_scheduler_smoke --slots 2: ok
cuda_v100_full_scheduler_smoke --slots 2: ok
```

Implementation note: the first version used `cuda_tmp_alloc` for the pointer
table and failed the full 8-stage smoke because that temp buffer is not
device-specific. The shipped version stores the pointer table in the
stage-owned scratch tensor and passed the full scheduler smoke.

Real replay correctness: first token id `926`, text `16`, hex `3136`.

## Sustained Decode Comparison

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | max GPU util |
|---|---:|---:|---:|---:|---:|
| Sprint 059 default-batched | 1 | 3.587298 | 3.363092 | 11.134% | 20.000% |
| Sprint 060 pointer-input | 1 | 3.564831 | 3.342029 | 10.935% | 20.000% |
| Sprint 059 default-batched | 2 | 3.862932 | 3.621499 | 11.434% | 20.000% |
| Sprint 060 pointer-input | 2 | 3.915266 | 3.670562 | 12.265% | 20.000% |

The pointer-input path improves two-slot generated tok/s by about `1.35%` over
Sprint 059 and about `5.69%` over Sprint 058. One-slot throughput is within
run-to-run noise and does not use the multi-slot pointer-input path.

Artifacts:

- `logs/from-cluster/sprint060-pointer-input-routed/replay.json`
- `logs/from-cluster/sprint060-pointer-input-routed/sustained_decode.tsv`
- `logs/from-cluster/sprint060-pointer-input-routed/sustained_decode.json`
- per-case `result.json`, status snapshots, `server.log`, and `gpu_util.csv`

## Assessment

Removing the per-slot routed input copy is correct and gives a small additional
speedup. It does not change the larger performance conclusion: two-slot decode
is still below `4` generated tok/s and average GPU utilization remains around
`12%`.

The next practical step should move from staging cleanup to kernel/runtime
shape: either batch the shared expert path, persist tensor views to cut CPU
overhead, or test higher active-slot/context tiers now that multi-slot layer
batching is the default path.
