# TEMP Status Report 018 - Sprint 204 Concurrent Resident TP4 Reduction

Date: 2026-05-23

## Topline

Sprint 204 added `DS4_TP4_RESIDENT_ALGO=doubling_async` to the resident TP4
layer-slice benchmark. This issues pairwise peer exchanges asynchronously on
each GPU's stream, then performs local add kernels before the next resident
layer.

Result:

- Correctness passes at 96 and 768 routes.
- Async doubling is much better than Sprint 203's sequential doubling.
- It is positive at larger 768-route shapes.
- It does not reliably clear the 96-route production decode gate.

## Current Best Served Baseline

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| `fused6_reduce + graph` | 256K | 16 | `67.886268` | `66.825545` | `16/16` |

## Sprint 204 Data

V100 target:

```text
test_ggml_turbomind_tp4_resident_layer_slice
```

Measured on GPUs `0,1,2,3`:

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

## Interpretation

This keeps TP4 alive only for larger batched/prefill shapes. It does not justify
production decode scheduler integration at the current 96-route 16-slot/256K
shape.

The next TP sprint would need a fused/NCCL-grade collective gate. If we do not
take that branch, the next practical-serving sprint should pivot to the
persistent fused routed-FFN executor.
