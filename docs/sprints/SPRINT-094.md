# Sprint 094: Grouped TurboMind And Shared F8 Serving

## Goal

Improve the production appliance hot path using the Sprint 093 profiler
evidence: reduce repeated TurboMind control-table churn and make routed expert
execution see the active slots as one grouped workload.

## Context

Sprint 093 fixed cold first-request serving by moving warmup into the server.
It also showed the 1M/4-slot appliance path was still only
`11.241074` generated tok/s and that the decode-window profile was led by F8
matmul launches, HtoD/control copies, and TurboMind GEMM.

Two code issues were visible:

- The packed TurboMind path rebuilt and uploaded per-expert device pointer
  tables every routed FFN call.
- The multi-slot FFN executor still called TurboMind once per slot instead of
  once per layer across all active slots.

## Implementation Plan

- Cache TurboMind packed matrix pointer tables per resident arena and tensor
  layout.
- Add a TurboMind batch pointer API that accepts per-slot hidden-state pointers
  and routes all active slots through one grouped expert call.
- Switch the multi-slot layer executor to the new TurboMind batch API.
- Re-test shared F8 FFN batching with the grouped TurboMind path and promote it
  to the launcher default if it wins.
- Validate on the V100 cluster with correctness, scheduler smoke, profiler
  evidence, and appliance soak throughput.

## Definition Of Done

- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [x] One-shot replay still returns first token hex `3136`.
- [x] Full scheduler smoke passes with `--slots 4`.
- [x] 4-request, 4-slot appliance soak passes with `warmup_requests=0`.
- [x] The default appliance launcher records `DS4_V100_BATCH_SHARED_F8=1`.
- [x] Sprint and vision docs record throughput and remaining bottlenecks.

## Result

Sprint 094 shipped a real, if still modest, production-path speedup.

Implementation:

- Added a resident TurboMind pointer-table cache keyed by arena pointer, GPU,
  packed tensor offsets, strides, shape, and expert count.
- Added
  `ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32`.
  It builds a per-slot hidden-state pointer table and calls the packed
  TurboMind grouped route implementation once for all active slots.
- Changed the batched FFN executor so TurboMind routed experts run once per
  layer batch rather than once per slot.
- Set `DS4_V100_BATCH_SHARED_F8=1` as the appliance launcher default after it
  measured faster with the grouped TurboMind routed path.

Validation:

```text
make tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
make tests/cuda_v100_full_scheduler_smoke CUDA_ARCH=sm_70 -j8
./tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 4
```

The full scheduler smoke passed:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=4 layers=43 tm_layers=43 ... ok
```

Throughput ladder at 1M context, 4 slots, 4 timed requests, 16 tokens/request,
and no client warmup:

| Build | Generated tok/s | Continuation tok/s | Notes |
|---|---:|---:|---|
| Sprint 093 startup warmup | `11.241074` | `10.538507` | Baseline |
| Pointer-table cache only | `11.925052` | `11.179736` | HtoD count dropped in profiler |
| Grouped TurboMind routed batch | `12.132966` | `11.374656` | Routed experts batched across slots |
| Grouped TurboMind + shared F8 default | `12.634955` | `11.845270` | Default after this sprint |

The final default soak:

```text
token_match=4/4
generated_tokens=64
continuation_tokens=60
elapsed_s=5.065313
latency_ms_avg=5019.078
aggregate_generated_tokens_per_second=12.634955
aggregate_continuation_tokens_per_second=11.845270
DS4_V100_BATCH_SHARED_F8=1
```

Profiler evidence:

- The table cache reduced the one-shot decode-window HtoD copy count from
  `801` to `153`.
- HtoD time stayed high because the remaining large copies are first-generation
  model-cache uploads in the diagnostic one-shot profile, not the small
  TurboMind pointer tables.
- The dominant non-copy bucket remains `arena_f8_e4m3_b128_matmul_kernel`.

The 8-slot/256K probe returned HTTP 200 and first-token hex `3136` for all
requests, but the soak harness split one request out of the tensor-batched
timing group and marked the run failed. Treat that as a harness/rendezvous
follow-up, not a passing throughput benchmark.

## Stop Conditions

- Stop if grouped TurboMind routing changes the expected first token.
- Stop if the full scheduler smoke fails at `--slots 4`.
- Stop if shared F8 batching regresses the final 4-slot appliance soak.
