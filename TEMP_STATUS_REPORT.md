# TEMP Status Report

Date: 2026-05-21

## Current State

We have a correct deployed 8x V100 DS4-Flash appliance path, but not yet a
practical high-throughput serving implementation. The best known topline is
still approximately:

| Mode | Context | Slots | Generated tok/s | Decode tok/s | Notes |
|---|---:|---:|---:|---:|---|
| Best short-context run | 16K | 256 | `61.223893` | `57.397400` | Sprint 146 control repeat |
| Best 32K run | 32K | 128 | `60.130047` | `56.371919` | Sprint 139 |
| Best 256K run | 256K | 16 | `46.394722` | `43.495052` | Sprint 128 opt-in stack |

This is still far below the practical serving vision target of roughly
`1k-2k` aggregate tok/s.

## What Changed In Sprint 156

The current focus is the fused routed-FFN software-pipeline path. Sprint 155
proved the stream-per-expert path was active but slower. I found one issue in
that test: the stream pipeline used eight stream groups while the profiled
served shape had six active expert groups.

Sprint 156 tested exact six-group scheduling:

| Shape | Mode | Generated tok/s | Decode tok/s | Correctness |
|---|---|---:|---:|---|
| 128-slot / 32K | control | `59.516392` | `55.796618` | 128/128 |
| 128-slot / 32K | stream pipeline, 6 groups | `59.645848` | `55.917982` | 128/128 |
| 256-slot / 16K | control | `60.442968` | `56.665283` | 256/256 |
| 256-slot / 16K | stream pipeline, 6 groups | `60.675527` | `56.883307` | 256/256 |

So exact-group pipelining is slightly positive in this deterministic benchmark:
about `+0.22%` at 128 slots and `+0.38%` at 256 slots. That is useful evidence,
but not enough to promote as a production default.

## Important Caveat

Hardcoding six stream groups is not generally safe. The test prompt/router path
shows six active experts, but real traffic may activate more experts. A
production-safe version needs either device-side dynamic compaction or a
fallback when active groups exceed the stream count.

I implemented an opt-in safe mode:

```text
DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=1
```

It reads the compacted route offsets, uses exactly the observed active group
count, and falls back if active groups exceed the stream limit. Correctness
passed, but served throughput regressed:

| Shape | Auto-group generated tok/s | Auto-group decode tok/s | Verdict |
|---|---:|---:|---|
| 128-slot / 32K | `58.988662` | `55.301871` | slower than control |
| 256-slot / 16K | `60.232265` | `56.467748` | slower than control |

The host-side active-group readback/sync costs more than the small stream-count
gain. This confirms that a host-orchestrated software pipeline is not the
material lever.

## Tensor Parallel Side Result

A parallel read-only analysis agrees with the previous TP evidence:

- Broad tensor parallelism is not a path to the `1k+` target.
- 2-way TP is only worth a bounded NV2-pair prototype for the 128-slot/32K
  768-route shape.
- It is no-go for single-slot decode and no-go for the current 256-slot/16K
  shape unless payload movement is eliminated or deeply overlapped.

## Current Decision

Do not promote the current group-pipeline path. Keep:

- `DS4_V100_TURBOMIND_GROUP_PIPELINE=1` as a diagnostic.
- `DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS=6` as a benchmark-only exact
  served-shape diagnostic.
- `DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=1` as a safe diagnostic, not
  a performance path.

The next real implementation target should be a persistent or larger fused
routed-FFN executor that avoids the host stream orchestration and repeated
launch/join overhead. The bounded 2-way TP prototype remains secondary.

## Latest Artifacts

- `docs/sprints/SPRINT-156.md`
- `logs/from-cluster/sprint156-fused-pipeline-stream-groups/`
- Changed but not yet committed: `ds4_cuda.cu`

## Remaining Gap

The appliance is functional and correct, but not performance-ready. We still
need a material routed-expert execution change, most likely:

1. persistent fused routed-FFN kernel/executor,
2. device-side active-expert compaction without host sync,
3. then, only if needed, narrow 2-way TP for the 128-slot/32K tier.
