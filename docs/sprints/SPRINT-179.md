# Sprint 179 - Compact Routed Executor No-Host-Sync Gate

Date: 2026-05-22
Status: Completed

## Overview

Sprint 178 showed that TP/EP host-thread overlap is not the missing throughput
lever. While reading the fused routed-FFN path, one remaining hot-path issue is
clear: when the routed executor and compact schedule are active, the runtime can
copy the full expert offset table back to the host to count active experts
before launching the compact TurboMind GEMMs.

For the production served shape this is unnecessary. `total_routes == 6`, so
the compact schedule can safely expose six compact groups and let empty groups
have equal start/end offsets. The existing `tm_build_compact_schedule_kernel`
already fills empty compact groups this way.

Sprint 179 adds a default-off gate:

```text
DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC=1
```

When enabled, and only for routed-executor compact scheduling with profiling
off and group-pipeline auto off, the code skips the device-to-host active-expert
read and keeps `tm_group_count = tm_group_capacity`.

## Non-Goals

- No promotion without served A/B improvement.
- No change to `DS4_V100_TURBOMIND_PROFILE=1`; profiling still reads the active
  expert histogram for visibility.
- No change to group-pipeline auto mode.
- No new TurboMind ABI.
- No persistent kernel rewrite.

## Architecture

Current fused6 compact path:

```text
build route offsets on GPU
copy offsets GPU -> CPU
count active experts on CPU
set compact group count
build compact pointer table on GPU
launch compact gate/up and down
```

Sprint 179 gated path:

```text
build route offsets on GPU
skip CPU active-expert read
use compact group capacity as group count
build compact pointer table on GPU
launch compact gate/up and down, including empty compact groups
```

For six-route decode, the maximum compact group count is six. This trades up to
six empty-group scheduler entries for removing a per-call synchronization.

## Implementation

- Add `DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC`.
- Keep the default path unchanged.
- In `ds4_cuda.cu`, gate the host active-expert read:
  - keep it for profile mode;
  - keep it for group-pipeline auto;
  - skip it for routed-executor compact schedule when the new flag is enabled.
- Add one-time verbose logging so the selected path is visible.
- Add launcher validation/export/startup recording.

## Validation

- Local build for `ds4_cuda.o` if supported.
- V100 build for:
  - `ds4_cuda.o`
  - `tools/ds4-v100-replay`
  - `tests/cuda_v100_full_scheduler_smoke`
  - `tests/cuda_v100_selected_token_smoke`
- V100 selected-token smoke with the gate enabled returns expected token `3136`.
- Served same-binary 16-slot/256K A/B:
  - control: production path with gate off;
  - candidate: `DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC=1`;
  - both must record prompt/generated/continuation tok/s and `16/16` token
    match.

## Files Summary

| File | Change |
|---|---|
| `ds4_cuda.cu` | Add guarded no-host-sync compact routed executor path |
| `tools/ds4-v100-run-appliance.sh` | Validate/export/log the new flag |
| `deploy/v100/ds4-v100-appliance.env.example` | Document default-off flag |
| `docs/sprints/VISION.md` | Record outcome |
| `logs/from-cluster/sprint179-compact-no-host-sync/` | V100 evidence |

## Definition Of Done

- [x] Default path unchanged when the gate is unset.
- [x] Launcher validates, exports, and logs the flag.
- [x] V100 build passes.
- [x] Selected-token smoke preserves expected token `3136`.
- [x] Served 16-slot/256K A/B is correct with `16/16` token match.
- [x] Promote only if continuation tok/s improves materially; otherwise keep
      diagnostic-only and use the result to choose the next persistent executor
      step.

## Results

Implemented `DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC=1` as a guarded path in
the compact routed executor. When the routed executor is active, compact
scheduling is enabled, profiling is off, and group-pipeline auto is off, the
runtime skips the device-to-host active-expert read and keeps
`tm_group_count = tm_group_capacity`.

V100 build on `llm/llamacpp-build-8gpu` passed:

```text
make ds4_cuda.o tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Correctness:

```text
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

The selected-token and served logs confirmed the new branch:

```text
ds4: TurboMind routed executor fused6_reduce no_host_sync total_routes=6 compact_groups=6
ds4: routed-FFN liveness executor=fused6_reduce total_routes=6 route_expanded_a_half=0 compact_a_half=1 gate_out=elided mid_half=materialized down_routes=elided output_mode=full_sum
ds4: TurboMind down-reduce epilogue selected total_routes=6
```

Served same-binary 16-slot/256K A/B, 16 requests x 64 generated tokens,
per-step async + event handoff:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| production control | `20.083371` | `71.407542` | `70.291799` | `16/16` |
| fused6_reduce host-sync | `19.730169` | `70.151713` | `69.055593` | `16/16` |
| fused6_reduce no-host-sync verbose | `19.210761` | `68.304927` | `67.237662` | `16/16` |
| fused6_reduce no-host-sync quiet | `19.151022` | `68.092522` | `67.028577` | `16/16` |

Decision: keep `DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC` default-off and
diagnostic-only. The skipped host sync was not the dominant bottleneck; running
empty compact groups costs more than the avoided synchronization for the
six-route served shape. This further narrows the path: the remaining issue is
not a small wrapper synchronization, but the routed-FFN boundary itself
(`mid_half` remains materialized). Next work should remove that materialization
or step back to a larger persistent executor design.

Evidence: `logs/from-cluster/sprint179-compact-no-host-sync/`.
