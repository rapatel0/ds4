# TEMP Status Report 382

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 382 hardened the resident serving startup path with VRAM
admission and telemetry so the `32` slot / `256K` appliance fails cleanly
before overfilling 32GB V100s.

This was triggered by Sprint 381: successful HTTP runs used roughly
`32418 MiB`, and one immediate E5M2 candidate startup hit CUDA OOM before
readiness.

## What Changed

- Added `--vram-report` and `--vram-min-free-mib N` to
  `tools/ds4-v100-tp-ep-full-layer-smoke`.
- Added planned dense-cache allocation checks before `cudaMalloc`.
- Added startup checkpoint rows:
  `tp_ep_vram_plan`, `tp_ep_vram`, and `tp_ep_vram_summary`.
- Added launcher env:
  `DS4_V100_TP_EP_VRAM_REPORT` and
  `DS4_V100_TP_EP_VRAM_MIN_FREE_MIB`.
- Added profile harness flags and `summary.json` parsing.

## V100 Validation

Artifacts:

```text
/workspace/logs/sprint382-vram/
```

Build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Passed on gpu-01.

Launcher proof:

```text
--vram-report --vram-min-free-mib 64
```

appears in the TP/EP appliance `--print-command` path when
`DS4_V100_TP_EP_VRAM_REPORT=1`.

Direct validation:

| Shape | Result |
|---|---|
| slots/context | `32` / `256K` |
| position | `262080` |
| decode steps | `1` |
| return code | `0` |
| first token | `54639` |
| generated decode tok/s | `66.136824` |
| minimum free VRAM | `1754 MiB` |
| maximum used VRAM | `30739 MiB` |
| VRAM failures | `0` |
| reserve threshold | `64 MiB` |

Negative threshold check:

```text
--vram-min-free-mib 40000
```

failed immediately at startup with:

```text
rc=14
tp_ep_vram_summary label startup min_free_mib 32182 max_used_mib 311 threshold_mib 40000 failures 8 FAIL
```

## Decision

Promote the VRAM telemetry/admission plumbing as default launcher hardening
with a `64 MiB` minimum free reserve.

This does not improve throughput directly. It removes an operational blind
spot so the next performance work can run with explicit memory-margin evidence.

## Next Best Work

Return to the active throughput thesis from `TEMP_THROUGHPUT_PROMPT.md`:

1. Re-baseline current TP/EP `32` slot / `256K` serving with GPU utilization
   and VRAM summaries enabled.
2. Focus the next implementation sprint on steady-state launch/sync overhead,
   especially host syncs and capture blockers in the typed attention/KV path.
3. Keep each optimization behind an isolated default-off gate and A/B it on
   gpu-01 before promotion.
