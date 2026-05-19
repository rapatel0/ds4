# Sprint 057 Report: Deterministic Token-Step Coalescing

## Result

`SHIP`, with the batched FFN layer slice gated off by default.

## Changes Implemented

1. Added a server-side microbatch rendezvous.
   - `tools/ds4-v100-replay.c` now broadcasts a pending condition variable on
     enqueue.
   - Batch processing waits briefly for peer requests when active microbatching
     is configured and MTP is off.
   - This fixes the race where the first handler entered `generation_mu` and
     drained a one-request batch before other concurrent requests arrived.
2. Added a batched MXFP4 routed route primitive.
   - `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_f32` indexes
     `[token][route]` selected experts and weights.
   - The existing single-token primitive remains as a wrapper.
3. Added an opt-in batched HC layer decode path.
   - `ds4_v100_layer_execute_hc_decode_batch` batches the routed FFN slice after
     per-slot attention/HC work.
   - `ds4_v100_stage_scheduler_decode_hc_batch` uses it only when
     `DS4_V100_BATCH_LAYER_FFN` is set.
4. Extended `tests/cuda_v100_mxfp4_moe_smoke.c`.
   - The smoke now validates both single-token grouped routes and the batched
     grouped route primitive.

## Validation

Local:

```bash
cc -fsyntax-only -I. tools/ds4-v100-replay.c
cc -fsyntax-only -I. ds4_v100_layer_execute.c
cc -fsyntax-only -I. ds4_v100_scheduler.c
cc -fsyntax-only -I. tests/cuda_v100_mxfp4_moe_smoke.c
make ds4_v100_layer_execute.o ds4_v100_scheduler.o tools/ds4-v100-replay.o tests/cuda_v100_mxfp4_moe_smoke.o
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

```text
cuda_v100_mxfp4_moe_smoke: ok
```

Real replay correctness: first token id `926`, text `16`, hex `3136`.

## Sustained Decode Comparison

Default path:

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | tensor batches |
|---|---:|---:|---:|---:|---:|
| Sprint 056 | 1 | 3.552642 | 3.330602 | 10.766% | 0 |
| Sprint 057 default | 1 | 3.560863 | 3.338309 | 10.655% | 0 |
| Sprint 056 | 2 | 3.676873 | 3.447068 | 10.964% | 0 |
| Sprint 057 default | 2 | 3.662490 | 3.433585 | 10.756% | 2 |

The default two-slot path now deterministically exercised token-step batching:
`tensor_batched_groups=2`, `tensor_batched_requests=4`, and
`tensor_batched_tokens=64`.

Opt-in batched FFN path:

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | tensor batches |
|---|---:|---:|---:|---:|---:|
| Sprint 057 opt-in `DS4_V100_BATCH_LAYER_FFN` | 1 | 3.552265 | 3.330249 | 10.546% | 0 |
| Sprint 057 opt-in `DS4_V100_BATCH_LAYER_FFN` | 2 | 3.630558 | 3.403648 | 10.969% | 2 |

The opt-in batched FFN slice is correct but slower than the default path, so it
remains disabled by default. The likely cause is that it only batches the routed
FFN slice after per-slot attention/HC work and pays extra tensor-copy/view
overhead without enough batch width to improve occupancy.

Artifacts:

- `logs/from-cluster/sprint057-coalescing-default/replay.json`
- `logs/from-cluster/sprint057-coalescing-default/sustained_decode.tsv`
- `logs/from-cluster/sprint057-coalescing-default/sustained_decode.json`
- `logs/from-cluster/sprint057-batched-ffn-optin/sustained_decode.tsv`
- `logs/from-cluster/sprint057-batched-ffn-optin/sustained_decode.json`

## Assessment

The sprint closes the measurement/coalescing gap from Sprint 056. Multi-slot
benchmark cases now reliably enter the existing batch API, which makes future
slot-scaling experiments meaningful.

It does not materially improve throughput. GPU utilization remains around
`11%`, and the first deeper batch slice regressed. The next sprint should either
reduce per-layer allocation/copy overhead in the opt-in batch path or move
directly to a persistent grouped MoE kernel that can amortize expert work across
many active tokens without the current per-slot setup cost.
