# Sprint 382: TP/EP VRAM Admission Margin

## Overview

Add permanent, low-overhead VRAM telemetry and admission checks to the TP/EP
serving path so the `32` slot / `256K` appliance fails before it overfills a
32GB V100.

This sprint is serving hardening, not a throughput optimization. It directly
addresses Sprint 381's startup OOM during E5M2 HTTP A/B and gives future
performance sprints a reliable memory-margin signal.

## Rationale

The current TP/EP serving shape is close to the physical V100 limit. Sprint
381 measured successful HTTP runs at `32418 MiB` max used and also saw an
immediate candidate run fail before readiness while allocating the dense cache.
That means the runtime can pass correctness and still be too fragile for
production operation.

Before more throughput work lands, the appliance needs:

- a planner-visible preallocation check for large resident arenas,
- checkpointed free/used VRAM telemetry during startup,
- launcher defaults that preserve a small explicit free-memory reserve,
- profile summaries that capture the minimum free VRAM seen during a run.

## Scope

- Add `--vram-report` and `--vram-min-free-mib N` to
  `tools/ds4-v100-tp-ep-full-layer-smoke`.
- Check planned dense F16 cache allocation before `cudaMalloc`.
- Emit `tp_ep_vram` and `tp_ep_vram_summary` rows at major resident startup
  checkpoints.
- Add launcher environment variables:
  `DS4_V100_TP_EP_VRAM_REPORT` and
  `DS4_V100_TP_EP_VRAM_MIN_FREE_MIB`.
- Add profile harness plumbing and summary parsing for VRAM fields.

## Out Of Scope

- No PP/layer-split work.
- No topology changes.
- No E5M2 promotion.
- No CUDA graph or kernel fusion changes.
- No automatic slot/context downshift yet; this sprint only reports and
  rejects unsafe startup shapes.

## Definition Of Done

- Local syntax checks pass:
  `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`,
  `bash -n tools/ds4-v100-run-appliance.sh`, and `git diff --check`.
- V100 build passes:
  `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Launcher `--print-command` proof shows `--vram-min-free-mib 64` by default
  and includes `--vram-report` when `DS4_V100_TP_EP_VRAM_REPORT=1`.
- A V100 direct or HTTP profile at `32` slots / `256K` with
  `--vram-report --vram-min-free-mib 64` records VRAM summary fields in
  `summary.json`.
- A high-threshold negative check fails cleanly with a VRAM admission or
  checkpoint failure before serving readiness.

## Risks

- A `64 MiB` reserve may be too tight if CUDA context overhead fluctuates. If
  V100 proof shows false failures, lower the default reserve and record the
  measured margin.
- `cudaMemGetInfo` is a point-in-time allocator signal, not a full production
  capacity planner. It is still stronger than allowing late CUDA OOMs during
  startup.
- Startup telemetry must remain out of the decode loop so it does not distort
  throughput measurements.

## Outcome

Implemented and validated as a permanent startup hardening feature.

Code changes:

- `tools/ds4-v100-tp-ep-full-layer-smoke` accepts
  `--vram-report` and `--vram-min-free-mib N`.
- The dense F16 cache path now checks planned per-GPU cache/temp bytes before
  allocation and emits `tp_ep_vram_plan` rows.
- Resident startup now emits `tp_ep_vram` and `tp_ep_vram_summary` rows at
  major allocation checkpoints.
- `tools/ds4-v100-run-appliance.sh` wires
  `DS4_V100_TP_EP_VRAM_REPORT` and
  `DS4_V100_TP_EP_VRAM_MIN_FREE_MIB`; the default reserve is `64 MiB`.
- `tools/ds4-v100-tp-ep-profile.py` passes the launcher/direct flags and
  records aggregate VRAM fields in `summary.json`.

V100 evidence:

| Check | Result |
|---|---|
| Local syntax | `python3 -m py_compile`, `bash -n`, and `git diff --check` passed |
| Build | `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke` passed |
| Launcher proof | `--print-command` includes `--vram-report --vram-min-free-mib 64` when reporting is enabled |
| Direct profile | `32` slots / `256K` / `position=262080` / `1` token returned `returncode=0` |
| VRAM summary | `vram_failures=0`, `vram_min_free_mib=1754`, `vram_max_used_mib=30739`, `vram_threshold_mib=64` |
| Decode sanity | first token `54639`, generated decode `66.136824` tok/s |
| Negative check | `--vram-min-free-mib 40000` fails at startup with `rc=14`, `failures=8`, no serving readiness attempt |

Artifacts:

```text
/workspace/logs/sprint382-vram/
```

## Decision

Promote the VRAM admission/telemetry plumbing as the TP/EP launcher default
with a `64 MiB` minimum free reserve.

This does not make the appliance production-ready by itself, and it does not
solve the throughput bottleneck. It does give every subsequent performance
sprint an explicit safety guard so failures are reported as memory-admission
events rather than late CUDA OOMs.
