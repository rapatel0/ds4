# Sprint 154 - Fused Routed-FFN Boundary Validation

Date: 2026-05-21

## Objective

Fully test the largest currently implemented fused routed-FFN boundary on the
V100 cluster before committing more work to software-pipeline variants:
TurboMind fused gate/up + gated-SiLU plus the DS4 down-projection route-weighted
reduce epilogue.

This sprint did not add a new kernel. It closed the missing served A/B for the
1536-route down-reduce path and re-tested the 768-route path with split
prefill/decode metrics.

## Served A/B

128-slot / 32K, 768 routed rows:

| Mode | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---|
| control | `59.502747` | `66.940590` | `55.783825` | 128/128 |
| down-reduce epilogue | `59.509317` | `66.947982` | `55.789985` | 128/128 |

256-slot / 16K, 1536 routed rows:

| Mode | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---|
| control | `60.671924` | `68.255915` | `56.879929` | 256/256 |
| down-reduce epilogue | `60.642962` | `68.223332` | `56.852777` | 256/256 |

The 128-slot result is run-noise flat. The 256-slot result is slightly slower.
Do not promote `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1`.

## Profile

Synchronized 128-slot / 32K full-scheduler profile:

| Mode | Gate/up share | Down share | Scatter/reduce shape |
|---|---:|---:|---|
| control | `~58-60%` on main stages | `~25-29%` | `~0.15-0.34 ms` per profiled stage aggregate |
| down-reduce epilogue | `~58-61%` on main stages | `~25-29%` | unchanged at the level that affects served throughput |

The fused epilogue does not materially change the dominant buckets. Gate/up and
down GEMM execution remain the bottleneck, not the final route-weighted reduce
tail.

## Tensor-Parallel Side Assessment

A parallel review of the TP evidence reached the same bounded conclusion as
Sprint 153: 2-way TP is worth at most a narrow one-layer prototype for the
128-slot/32K NV2-pair shape. It is not a broad replacement for layer sharding
and it does not explain the gap to the practical `1k+` aggregate tok/s target.

Back-of-envelope end-to-end speedup from the measured TP proxy is only about
`1.1-1.2x` if the routed FFN dominates the full decode wall time. The
256-slot/16K shape is already copy-limited in the existing proxy.

## Decision

Stage-count tuning inside gate/up and epilogue-only fusion are now both ruled
out as material software-pipeline levers.

The next implementation must change the actual routed expert execution model:

- a DS4-only persistent/grouped routed-FFN executor that keeps expert work
  resident across gate/up, activation, down, and weighted accumulation; or
- a narrow one-layer 2-way TP executor for the 128-slot/32K tier, used only as
  a bounded scheduling experiment.

## Artifacts

- `logs/from-cluster/sprint154-fused-pipeline-ab/`
- `logs/from-cluster/sprint154-fused-pipeline-profile/`

## Validation

- `tools/ds4-v100-appliance-soak.sh` at 128-slot / 32K control and
  down-reduce candidate.
- `tools/ds4-v100-appliance-soak.sh` at 256-slot / 16K control and
  down-reduce candidate.
- `tests/cuda_v100_full_scheduler_smoke` with
  `DS4_V100_TURBOMIND_PROFILE=1` for 128-slot / 32K control and down-reduce
  candidate.
