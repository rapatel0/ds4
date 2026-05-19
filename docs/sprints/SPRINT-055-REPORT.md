# Sprint 055 Report: Fused MXFP4 Routed Down Accumulation

## Result

`SHIP`.

## Changes Implemented

1. Added `ds4_gpu_arena_mxfp4_matmul_add_f32`.
   - Decodes a source MXFP4 matrix row and accumulates directly into an output
     vector with `out = add + matmul(row, x)`.
   - Uses the same MXFP4 row validation and nibble/scale semantics as the
     existing source MXFP4 matmul.
2. Updated `execute_ffn_delta`.
   - Routed down projection and route accumulation are now one CUDA primitive.
   - The per-route loop remains; this sprint removes one launch per selected
     route but does not yet group the six routes.
3. Extended `tests/cuda_v100_mxfp4_moe_smoke.c`.
   - The smoke compares fused down accumulation against the previous separate
     down matmul plus add path.

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
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_ARCH=sm_70 make tests/cuda_v100_mxfp4_moe_smoke tools/ds4-v100-replay
'
```

Focused V100 smoke:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_VISIBLE_DEVICES=0 ./tests/cuda_v100_mxfp4_moe_smoke
'
```

Result:

```text
cuda_v100_mxfp4_moe_smoke: ok
```

Real replay correctness: first token id `926`, text `16`, hex `3136`.

## Sustained Decode Comparison

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | max GPU util |
|---|---:|---:|---:|---:|---:|
| Sprint 054 | 1 | 3.384749 | 3.173202 | 10.679% | 21.000% |
| Sprint 055 | 1 | 3.410425 | 3.197274 | 10.893% | 21.000% |
| Sprint 054 | 2 | 3.486851 | 3.268923 | 11.157% | 22.000% |
| Sprint 055 | 2 | 3.503283 | 3.284328 | 11.386% | 21.000% |

Artifacts:

- `logs/from-cluster/sprint055-mxfp4-down-accum/replay.json`
- `logs/from-cluster/sprint055-mxfp4-down-accum/sustained_decode.tsv`
- `logs/from-cluster/sprint055-mxfp4-down-accum/sustained_decode.json`
- per-case `result.json`, `server_status_before.json`,
  `server_status_after.json`, `server.log`, and `gpu_util.csv`

## Assessment

The fused down+accumulation primitive is correct and gives another small
end-to-end gain: about `0.8%` in the one-slot sustained run and about `0.5%` in
the two-slot sustained run over Sprint 054.

The result is useful mostly as a stopping signal for one-route-at-a-time
cleanup. We have removed several launches per route, but GPU utilization
remains near `11%`. The next meaningful optimization needs to group selected
routes or batch layer execution across slots so each launch does substantially
more work.
