# Sprint 054 Report: Fused MXFP4 Routed Gate-Up Kernel

## Result

`SHIP`.

## Changes Implemented

1. Added `ds4_gpu_arena_mxfp4_pair_swiglu_f32`.
   - Decodes source MXFP4 gate and up rows in one kernel.
   - Reuses the same E8M0 scale and MXFP4 nibble decode semantics as
     `ds4_gpu_arena_mxfp4_matmul_f32`.
   - Applies the same clamp and SwiGLU math as `ds4_gpu_swiglu_tensor`.
2. Updated `execute_ffn_delta`.
   - Each routed expert now uses the fused gate+up+SwiGLU primitive.
   - The down projection and accumulation remain unchanged.
   - Standalone MXFP4 matmul remains available and is still used for down.
3. Extended `tests/cuda_v100_mxfp4_moe_smoke.c`.
   - The smoke computes the previous separate gate/up/SwiGLU path and compares
     it with the fused primitive before using the fused mid vector for the down
     projection.

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

Real replay correctness:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./tools/ds4-v100-replay \
    --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
    --model /models/DSv4-Flash-256e-fixed.gguf \
    --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
    --tokens 2 \
    --expected-token-hex 3136 \
    --json > logs/sprint054-fused-mxfp4-replay.json
'
```

Result: first token id `926`, text `16`, hex `3136`.

## Sustained Decode Comparison

Executed on `llamacpp-build-8gpu`:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  timeout 1200 bash ./tools/ds4-v100-sustained-decode-bench.sh \
    --model /models/DSv4-Flash-256e-fixed.gguf \
    --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
    --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
    --ctx-tiers 1048576 \
    --slot-tiers 1,2 \
    --queue-policies sequential \
    --tokens 16 \
    --requests 4 \
    --warmup-requests 1 \
    --expected-token-hex 3136 \
    --sample-ms 500 \
    --log-dir logs/sprint054-fused-mxfp4
'
```

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | max GPU util |
|---|---:|---:|---:|---:|---:|
| Sprint 053 | 1 | 3.291466 | 3.085749 | 10.768% | 22.000% |
| Sprint 054 | 1 | 3.384749 | 3.173202 | 10.679% | 21.000% |
| Sprint 053 | 2 | 3.371659 | 3.160931 | 11.133% | 22.000% |
| Sprint 054 | 2 | 3.486851 | 3.268923 | 11.157% | 22.000% |

Artifacts:

- `logs/from-cluster/sprint054-fused-mxfp4/replay.json`
- `logs/from-cluster/sprint054-fused-mxfp4/sustained_decode.tsv`
- `logs/from-cluster/sprint054-fused-mxfp4/sustained_decode.json`
- per-case `result.json`, `server_status_before.json`,
  `server_status_after.json`, `server.log`, and `gpu_util.csv`

## Assessment

The fused routed gate+up+SwiGLU primitive is correct and gives a small but real
end-to-end speedup: about `2.8%` for the one-slot sustained run and about
`3.4%` for the two-slot sustained run compared with Sprint 053.

This is not enough to materially change practical serving throughput. GPU
utilization remains around `11%`, and the result supports the next step:
continue fusing the routed FFN path, especially down projection plus route
accumulation and grouping multiple selected routes into fewer launches.
