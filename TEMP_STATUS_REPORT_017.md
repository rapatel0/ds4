# TEMP Status Report 017 - Sprint 203 Resident TP4 Layer Slice

Date: 2026-05-23

## Topline

Sprint 203 built the missing resident TP4 layer-slice gate. It combines the
real TurboMind MXFP4 TP4 routed-FFN split with a device-resident hidden-state
reduction loop across multiple layers.

Result:

- Correctness passes.
- Naive resident root reduction is still slower than one-GPU full-width routed
  FFN at 96 and 768 routes.
- The simple hand-rolled doubling variant is slower than root in this benchmark.
- Do not wire this TP4 boundary into production until there is a real concurrent
  collective/fused reduction boundary.

## Current Best Served Baseline

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| `fused6_reduce + graph` | 256K | 16 | `67.886268` | `66.825545` | `16/16` |

## Sprint 203 Data

V100 target:

```text
test_ggml_turbomind_tp4_resident_layer_slice
```

Measured on GPUs `0,1,2,3`:

| Shape | Algo | Full ref | Resident TP4 | Speedup | Boundary/iter | Correctness |
|---|---|---:|---:|---:|---:|---|
| `6 routes x 1 layer` | root | `0.1591 ms` | `0.1726 ms` | `0.921x` | `0.28 MiB` | PASS |
| `96 routes x 4 layers` | root | `1.1625 ms` | `1.4949 ms` | `0.778x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | root | `4.3276 ms` | `7.0362 ms` | `0.615x` | `144.00 MiB` | PASS |
| `96 routes x 4 layers` | doubling | `1.2657 ms` | `2.1035 ms` | `0.602x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | doubling | `4.7441 ms` | `9.4932 ms` | `0.500x` | `144.00 MiB` | PASS |
| `96 routes x 43 layers` | root | `12.9784 ms` | `15.7303 ms` | `0.825x` | `193.50 MiB` | PASS |
| `768 routes x 43 layers` | root | `44.0628 ms` | `74.8701 ms` | `0.589x` | `1548.00 MiB` | PASS |

Evidence:

```text
logs/from-cluster/sprint203-tp4-resident-layer-slice/
```

## Interpretation

Sprint 202 showed the TP4 expert compute is attractive when measured alone.
Sprint 203 shows the first resident boundary implementation does not preserve
that advantage. The culprit is now sharper: not routed-only host/device copies,
but the hidden-state all-reduce boundary itself.

The next TP implementation should not touch the production scheduler unless it
first provides a real concurrent collective or a fused reduction boundary that
beats this root baseline. Otherwise, the practical serving path should pivot
back to a persistent/fused routed-FFN executor on each layer-owned GPU.
