# Sprint 056 Report: Grouped MXFP4 Selected-Route Execution

## Result

`SHIP`.

## Changes Implemented

1. Added `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32`.
   - Processes all selected routed MXFP4 experts using router-selected expert
     ids and route weights.
   - Runs grouped gate/up/SwiGLU into `[routes x mid]` scratch.
   - Runs grouped routed down projection and sums routes into the routed FFN
     output.
2. Updated `execute_ffn_delta`.
   - The main FFN routed path now dispatches the grouped primitive instead of
     looping six route launch pairs.
   - The shared F8 expert path is unchanged.
3. Extended `tests/cuda_v100_mxfp4_moe_smoke.c`.
   - The smoke still validates the per-route fused primitives.
   - It now also compares grouped-route output against the per-route
     accumulation reference.

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
| Sprint 055 | 1 | 3.410425 | 3.197274 | 10.893% | 21.000% |
| Sprint 056 | 1 | 3.552642 | 3.330602 | 10.766% | 20.000% |
| Sprint 055 | 2 | 3.503283 | 3.284328 | 11.386% | 21.000% |
| Sprint 056 | 2 | 3.676873 | 3.447068 | 10.964% | 20.000% |

Sprint 056 improved generated tok/s by about `4.17%` at one slot and about
`4.96%` at two slots over Sprint 055. GPU utilization remains near `11%`, so
the dominant practical-use gap is still underfilled kernels and scheduler shape,
not model residency or correctness.

The two-slot benchmark status still reported `tensor_batched_groups=0`, so the
observed gain is attributable to grouped routed execution rather than request
coalescing.

Artifacts:

- `logs/from-cluster/sprint056-grouped-mxfp4-routes/replay.json`
- `logs/from-cluster/sprint056-grouped-mxfp4-routes/sustained_decode.tsv`
- `logs/from-cluster/sprint056-grouped-mxfp4-routes/sustained_decode.json`
- per-case `result.json`, `server_status_before.json`,
  `server_status_after.json`, `server.log`, and `gpu_util.csv`

## Assessment

Grouping selected routes gives a larger end-to-end uplift than the previous
single-route launch fusions, while preserving selected-token correctness. It is
still not close to the practical serving target because the grouped primitive is
not a persistent or tensor-core-friendly MoE kernel. The next sprint should move
up one level of granularity: either deterministic token-step coalescing in the
benchmark/request loop or a true batched layer executor that makes multiple
slots visible to the expensive kernels.
