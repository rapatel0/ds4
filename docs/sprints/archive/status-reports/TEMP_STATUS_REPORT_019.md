# TEMP Status Report 019 - Sprint 205 Async Root TP4 Reduction

Date: 2026-05-23

## Topline

Sprint 205 tested `DS4_TP4_RESIDENT_ALGO=root_async`, a concurrent root
gather/reduce/broadcast path for the resident TP4 layer-slice benchmark.

Result:

- Correctness passes.
- `root_async` is slower than one GPU at 96 and 768 routes.
- It is also slower than Sprint 204 `doubling_async`.
- The current TP4 decode branch should pause; next sprint should pivot to the
  persistent fused routed-FFN executor.

## Current Best Served Baseline

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| `fused6_reduce + graph` | 256K | 16 | `67.886268` | `66.825545` | `16/16` |

## Sprint 205 Data

V100 target:

```text
test_ggml_turbomind_tp4_resident_layer_slice
```

Measured on GPUs `0,1,2,3`:

| Shape | Algo | Full ref | Resident TP4 | Speedup | Boundary/iter | Correctness |
|---|---|---:|---:|---:|---:|---|
| `96 routes x 4 layers` | root_async | `1.2683 ms` | `1.3071 ms` | `0.970x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | root_async | `4.7405 ms` | `5.4744 ms` | `0.866x` | `144.00 MiB` | PASS |
| `96 routes x 43 layers` | root_async | `11.8966 ms` | `13.8286 ms` | `0.860x` | `193.50 MiB` | PASS |

Evidence:

```text
logs/from-cluster/sprint205-tp4-root-async/
```

## Interpretation

This gives a clean TP4 decode stop condition. The only positive TP4 result in
the current branch is the larger 768-route `doubling_async` shape from Sprint
204. The production 96-route 16-slot/256K shape remains negative.

Next implementation should pivot to persistent fused routed-FFN work rather
than another TP4 scheduler integration attempt.
