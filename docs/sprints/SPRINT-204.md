# Sprint 204 - Concurrent Resident TP4 Reduction Gate

Date: 2026-05-23
Status: Completed

## Objective

Upgrade the Sprint 203 resident TP4 layer-slice benchmark with a concurrent
four-GPU reduction path and measure whether the TP4 boundary becomes competitive
when peer exchanges are issued asynchronously per device.

## Rationale

Sprint 203 proved the resident TP4 routed-FFN layer slice is correct, but the
naive root boundary and sequential hand-rolled doubling both lose to the
one-GPU full-width reference. That result should not immediately kill TP4,
because the tested doubling path serialized peer exchanges and add kernels.

Before pivoting away from TP4, test the smallest materially different
collective shape: pairwise doubling where each device issues its peer copy and
local add on its own stream, allowing the V100 NVLink island to overlap the
four directions.

## Scope

1. Add `DS4_TP4_RESIDENT_ALGO=doubling_async`.
2. Reuse the same resident TP4 layer-slice benchmark and correctness gate.
3. Keep all data device-resident.
4. Measure the same 96-route and 768-route layer-slice shapes.
5. Compare against Sprint 203 root and sequential doubling results.

## Non-Goals

- No production scheduler integration.
- No NCCL dependency yet.
- No custom fused all-reduce kernel beyond per-device async peer copy plus
  local add.

## Definition Of Done

- [x] Sprint plan exists.
- [x] `doubling_async` builds in `test_ggml_turbomind_tp4_resident_layer_slice`.
- [x] V100 correctness passes at 96 and 768 routes.
- [x] Benchmark reports whether async doubling beats root and sequential
      doubling.
- [x] Evidence is copied into
      `logs/from-cluster/sprint204-tp4-resident-reduction/`.
- [x] Status, vision, experiment, and TEMP report documents are updated.
- [x] Changes are committed.

## Decision Gate

If async doubling is still slower than root or still loses badly to the
one-GPU reference, pause TP4 scheduler work and pivot the next sprint to a
persistent fused routed-FFN executor.

If async doubling materially improves the 43-layer 96-route resident slice,
continue TP4 with a repo-owned fused/NCCL-grade collective gate before any
runtime integration.

## Implementation

Added `DS4_TP4_RESIDENT_ALGO=doubling_async` to
`test_ggml_turbomind_tp4_resident_layer_slice`. The new path keeps the same TP4
routed-FFN layer loop but changes the resident reduction:

1. round 1 exchanges `0<->1` and `2<->3`;
2. each GPU issues its peer copy with `cudaMemcpyPeerAsync` on its own stream;
3. each GPU adds the peer payload locally on the same stream;
4. round 2 exchanges `0<->2` and `1<->3`;
5. the reduced hidden output becomes the next layer input by swapping resident
   buffers.

## Validation

V100 build passed:

```text
cmake --build build/turbomind-v100 --target test_ggml_turbomind_tp4_resident_layer_slice -j80
```

Measured on devices `0,1,2,3`:

| Shape | Algo | Full ref | Resident TP4 | Speedup | Boundary/iter | Correctness |
|---|---|---:|---:|---:|---:|---|
| `96 routes x 4 layers` | doubling_async | `1.2666 ms` | `1.1977 ms` | `1.058x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | doubling_async | `4.7399 ms` | `3.9946 ms` | `1.187x` | `144.00 MiB` | PASS |
| `96 routes x 43 layers` | doubling_async | `12.9856 ms` | `12.9090 ms` | `1.006x` | `193.50 MiB` | PASS |
| `96 routes x 43 layers repeat` | doubling_async | `11.1238 ms` | `12.4116 ms` | `0.896x` | `193.50 MiB` | PASS |
| `768 routes x 43 layers` | doubling_async | `46.0434 ms` | `43.0057 ms` | `1.071x` | `1548.00 MiB` | PASS |

Evidence:

```text
logs/from-cluster/sprint204-tp4-resident-reduction/
```

## Decision

Async doubling is a real improvement over Sprint 203's root and sequential
doubling reductions, but it does not clear the production 96-route decode gate.

At 768 routes the resident TP4 slice is positive (`1.071x` over 43 layers),
which keeps TP4 plausible for larger batched/prefill shapes. At the production
96-route shape, the first 43-layer result was only noise-positive (`1.006x`)
and the longer repeat was slower (`0.896x`). Do not integrate TP4 into the
production decode scheduler yet.

Next step: either build a fused/NCCL-grade collective gate for the resident
slice, or pivot to the persistent fused routed-FFN executor as the higher
probability practical-serving lever.
