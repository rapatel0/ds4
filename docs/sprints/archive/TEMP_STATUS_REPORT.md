# TEMP Status Report

Date: 2026-05-21

## Executive Status

The 8x V100 DS4-Flash appliance path is still correct and deployed, but it is
not performance-ready for practical high-throughput serving. The best observed
aggregate throughput remains about `61` generated tok/s. The practical vision
target remains roughly `1k-2k` aggregate tok/s, so the remaining gap is still
the routed expert execution model.

Latest committed checkpoint before Sprint 158:

```text
7640fcc turbomind: add graph replay diagnostic
```

Current Sprint 158 work adds an opt-in routed-FFN executor selector:
`DS4_V100_TURBOMIND_ROUTED_EXECUTOR=off|auto|fixed96|fixed768`. The implemented
`fixed96` path is deliberately guarded: it only runs when the compacted shape
is legal for the existing TurboMind 96-route fixed kernel.

Key result: the full scheduler can present the intended 16-slot/256K shape
(`total_routes=96`, six active compact groups), and the fixed96 gate_up kernel
is selected and correct. The HTTP served path, however, still reaches the FFN
as one request at a time (`total_routes=6`), so the fixed96 kernel is not
selected during the real served A/B. This makes request coalescing / served
batch formation the next blocking issue before fused 96-route kernels can move
production throughput.

## Current Topline

| Mode | Context | Slots | Generated tok/s | Decode tok/s | Status |
|---|---:|---:|---:|---:|---|
| Best short-context run | 16K | 256 | `61.223893` | `57.397400` | Correct, Sprint 146 control repeat |
| Best 32K run | 32K | 128 | `60.130047` | `56.371919` | Correct, Sprint 139 |
| Best 64K run | 64K | 64 | `57.322945` | `53.740261` | Correct, Sprint 136 |
| Best 128K run | 128K | 32 | `52.840889` | `49.538334` | Correct, Sprint 135 |
| Best 256K run | 256K | 16 | `46.394722` | `43.495052` | Correct, Sprint 128 opt-in stack |
| Sprint 158 served control | 256K | 16 | `46.113721` | `43.231614` | Correct, 16/16 |
| Sprint 158 fixed96 guarded | 256K | 16 | `46.167311` | `43.281854` | Correct, 16/16, fixed96 not selected in served path |
| Best 1M run | 1M | 4 | `21.771077` | `20.410385` | Correct, Sprint 119 |

## Sprint 158 Status

Implemented:

- `DS4_V100_TURBOMIND_ROUTED_EXECUTOR`.
- `DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE`.
- TurboMind ABI lookup for `ggml_turbomind_ds4_mxfp4_gated_silu_96`.
- Guarded fixed96 dispatch after compact active-expert inspection.
- Shape guard tightened so served single-request `total_routes=6` pays no
  active-group readback and falls through to the existing generic path.

Validation:

| Test | Result |
|---|---|
| V100 build: `ds4_cuda.o`, `tools/ds4-v100-replay`, `tests/cuda_v100_full_scheduler_smoke` | Pass |
| Launcher `--check` with `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed96` | Pass |
| Full 43-layer 16-slot/256K smoke, executor off | Pass |
| Full 43-layer 16-slot/256K smoke, fixed96 | Pass, selected fixed gate_up |
| Served 16-slot/256K control | `46.113721` generated / `43.231614` decode tok/s |
| Served 16-slot/256K fixed96 guarded | `46.167311` generated / `43.281854` decode tok/s |

Important diagnostic:

```text
full scheduler: total_routes=96 active_experts=6 max_routes_per_expert=16
served HTTP:    total_routes=6  active_experts=6 max_routes_per_expert=1
```

## Sprint 157 Status

Implemented:

- `DS4_V100_TURBOMIND_GRAPH=1` experimental routed-FFN CUDA Graph replay.
- `DS4_V100_TURBOMIND_GRAPH_VERBOSE=1` capture diagnostics.
- Graph cache keyed by GPU, routed shape, relevant TurboMind offsets, tensor
  pointers, mode flags, and accumulation mode.
- Warmup-before-capture behavior so the existing scratch/table caches are
  populated before capture.
- Forced no-readback total-token TurboMind ABI during capture.
- Single-slot served scratch reuse when graph replay is enabled, so the served
  path no longer churns fresh per-token tensor pointers before graph capture.

Validation so far:

| Test | Result |
|---|---|
| V100 build after graph implementation | Pass |
| V100 build after single-slot scratch change | Pass |
| Full 43-layer scheduler smoke, graph off | Pass |
| Full 43-layer scheduler smoke, graph on | Pass, emits warmup keys |
| Served 128-slot/32K graph candidate, global capture | Correct, but no captures |
| Served 128-slot/32K graph candidate, thread-local capture | Correct, but no captures |
| Small async-off diagnostic | Correct, but no captures |

Served 128-slot/32K graph measurements:

| Run | Generated tok/s | Decode tok/s | Graph captures | Notes |
|---|---:|---:|---:|---|
| Recent control | `59.607704` | `55.882222` | n/a | Graph disabled |
| Graph + stable scratch, global capture | `59.450666` | `55.734999` | `0` | 43 begin-capture failures |
| Graph + stable scratch, thread-local capture | `59.367233` | `55.656781` | `0` | 43 begin-capture failures |

The CUDA error is consistently:

```text
operation not permitted when stream is capturing
```

This happens even when the soak is forced to `--async-pipeline-mode off`, so my
current read is that graph capture around this path is incompatible with the
legacy default-stream kernel launch structure. To make CUDA Graph replay real,
we would need to thread an explicit capture/launch stream through the
TurboMind routed-FFN kernels and associated CUDA calls. That is a larger change
than the Sprint 157 wrapper-level probe.

## What This Means

The graph probe is useful evidence, but it is not a performance win:

- It did not capture in served mode.
- The fallback path is slightly slower because it pays graph-key/cache checks.
- It confirms launch replay alone is not available without explicit stream
  plumbing.

I would not promote `DS4_V100_TURBOMIND_GRAPH`. It should remain default-off
and diagnostic-only unless we decide to do the explicit-stream rewrite.

## Current Bottleneck

The high-slot routed-FFN profile still points at the expert GEMMs:

| Bucket | Approx share of profiled routed FFN time |
|---|---:|
| Gate/up MXFP4 GEMM | `~61%` |
| Down MXFP4 GEMM | `~31%` |
| Route build/gather/scatter/reduce combined | `~8%` |

This explains why small scheduler, tail, graph-wrapper, or stream-wrapper
tweaks are not moving the headline. We need to change the routed expert
execution model itself.

## Recommended Next Step

Stop treating wrapper-level CUDA Graph replay as the main path. The next
implementation should be one of:

- explicit-stream routed-FFN executor if we still want graph replay; or
- true persistent/larger fused routed-FFN kernel covering gate/up, activation,
  down, and route-weighted reduce; or
- bounded 2-way TP routed-FFN prototype for the 128-slot/32K NV2 case.

Given the latest evidence, the persistent/larger fused routed-FFN boundary is
still the most direct continuation of the current kernel work.
