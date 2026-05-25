---
sprint: 346
title: TP/EP Decode Profiler Window
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 346 - TP/EP Decode Profiler Window

## Overview

Sprint 345 proved that the typed TP/EP serving path is not simply missing
tensor-core kernels. Cutlass WMMA/HMMA and TurboMind SM70 FP4 HMMA are both
present, but a tiny 64-token serving profile still produced roughly `274k`
kernel launches and `178k` memcpy events. The next step is to collect a clean
decode-only profile window and make that mechanism permanent, so optimization
work can target the actual hot region instead of startup, HTTP lifecycle, or
profiler shutdown artifacts.

This sprint is a TP/EP-only sprint. No PP/layer-split work. No MTP work.

## Goals

- Add `cudaProfilerStart/Stop` support to the TP/EP full-layer smoke/server
  binary, not only the frozen replay path.
- Keep profiler control opt-in and zero-overhead in normal serving.
- Extend the profiler harness with windowed profiler modes that use
  `--profile-from-start off` when available.
- Validate a decode-only window on the V100 pod at the target serving shape:
  `32` slots, `256K` context, typed production KV enabled.
- Use the clean trace to decide the next implementation target for fusion or
  boundary removal.

## Implementation

1. Add `Options::cuda_profiler_window` to
   `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
2. Parse `--cuda-profiler-window` and show it in usage.
3. Wrap direct `run_token_major_serving_loop` execution with profiler
   start/stop.
4. Wrap each HTTP generation/prefill batch decode window with profiler
   start/stop, without profiling server startup or teardown.
5. Add permanent harness modes to `tools/ds4-v100-tp-ep-profile.py`:
   - `nvprof-window-gpu-trace`
   - `nvprof-window-api-trace`
   - `ncu-window-basic`
   - `ncu-window-nvlink`
6. Set `DS4_V100_CUDA_PROFILER_WINDOW=1` only for windowed modes.
7. Preserve the existing non-window modes for broad trace capture.

## Verification

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run at least:

```text
32 slots
32 concurrent requests
256K context
typed KV history + skip-current-load + quiet + batch-rows + stream-sync
tokens/request: 1 or 2
```

Required artifacts:

- no-profiler sanity summary
- windowed `nvprof` GPU trace summary
- parsed top-kernel TSV for the windowed trace
- if `ncu` emits metrics cleanly, keep the NCU report summary; if not, record
  why and keep the command/output evidence

## Definition of Done

- [x] TP/EP binary accepts `--cuda-profiler-window`.
- [x] Normal TP/EP serving path remains unchanged when the flag is off.
- [x] Windowed profiler modes are available in the permanent harness.
- [x] V100 build passes.
- [x] V100 windowed profile run returns correct HTTP responses.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] `docs/sprints/STATUS.md` and `TEMP_STATUS_REPORT_058.md` summarize the
      result and the next optimization target.
- [x] Sprint artifacts are committed.

## Expected Decision

If the windowed trace matches Sprint 345, the next sprint should implement the
first fusion/boundary-removal target around the largest non-GEMM families:
HC-current gather/fill, dense input fill/gather, and compressor staging. If
the windowed trace materially changes the ranking, use the windowed ranking as
the optimization contract.

## Outcome

Implemented:

- `--cuda-profiler-window` in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- TP/EP launcher propagation for `DS4_V100_CUDA_PROFILER_WINDOW=1`
- windowed profiler modes in `tools/ds4-v100-tp-ep-profile.py`

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

No-profiler sanity at `32` slots / `256K` / `32` requests / `2` tokens:

```text
http_200: 32/32
coalesced_batch_size: 32
server_generated_tok_s: 78.032873
server_generated_tok_s_decode: 92.070787
```

Windowed `nvprof` run:

```text
http_200: 32/32
coalesced_batch_size: 32
server_generated_tok_s: 72.078755
server_generated_tok_s_decode: 84.920155
profiler window markers: 36
gpu trace bytes: 0
```

Windowed `ncu` run with `regex:.*cutlass_70_wmma.*` and launch count `8`:

```text
http_200: 32/32
coalesced_batch_size: 32
server_generated_tok_s: 35.193444
server_generated_tok_s_decode: 39.720168
profiler window markers: 34
report: lifecycle lines only; no kernel metrics
```

The implementation is correct and useful, but the profiler result is negative:
CUDA profiler start/stop markers fire inside the TP/EP decode path, yet
`nvprof --profile-from-start off` writes an empty GPU trace and Nsight Compute
still emits only process lifecycle output in this HTTP-wrapper setup. The next
sprint should add a direct non-server TP/EP replay/profile target that exits
naturally after one decode window. That target should reuse the same resident
state and 32-slot typed decode path but avoid the HTTP server lifecycle and
post-request process termination interaction.

## Artifacts

- `logs/from-cluster/sprint346-tp-ep-profiler-window/cluster/none/summary.json`
- `logs/from-cluster/sprint346-tp-ep-profiler-window/cluster/nvprof-window-gpu-trace/summary.json`
- `logs/from-cluster/sprint346-tp-ep-profiler-window/cluster/nvprof-window-gpu-trace/server.err`
- `logs/from-cluster/sprint346-tp-ep-profiler-window/cluster/ncu-window-basic/ncu-window-basic.csv`
- `logs/from-cluster/sprint346-tp-ep-profiler-window/cluster/ncu-window-basic/server.err`
