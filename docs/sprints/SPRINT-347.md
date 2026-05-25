---
sprint: 347
title: Direct TP/EP Profile Target
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 347 - Direct TP/EP Profile Target

## Overview

Sprint 346 wired CUDA profiler windows into TP/EP serving, but the HTTP-wrapper
process still did not produce scoped `nvprof` or Nsight Compute metrics. The
next step is to remove HTTP from the profiler experiment. The TP/EP binary
already has a resident token-major all-layer mode that exits naturally after a
fixed decode window. This sprint makes that mode available from the permanent
profiler harness.

This is TP/EP-only work. No PP/layer-split work. No MTP.

## Goals

- Add a direct non-server run mode to `tools/ds4-v100-tp-ep-profile.py`.
- Reuse the same resident 32-slot typed KV decode flags as the HTTP serving
  path.
- Preserve the existing HTTP profiler harness behavior.
- Capture direct no-profiler and windowed profiler artifacts on the V100 pod.
- Decide whether direct profiler mode is enough for clean NCU/nvprof metrics.

## Implementation

1. Add `--run-mode http|direct-token-major` to the profiler harness.
2. Build the direct command from explicit
   `tools/ds4-v100-tp-ep-full-layer-smoke` arguments instead of the HTTP
   launcher.
3. Save direct `stdout`, `stderr`, command, summary, and parsed top-kernel
   output into the same artifact tree.
4. Parse `tp_ep_serving_bench`, `tp_ep_token_major_scaffold`, and profiler
   marker counts from direct output.

## Verification

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100:

```text
direct-token-major, no profiler
direct-token-major, nvprof-window-gpu-trace
```

Target shape:

```text
slots: 32
ctx: 256K
decode steps: 1-2
typed production KV: history + skip-current-load + quiet + batch-rows + stream-sync
```

## Definition of Done

- [x] `--run-mode direct-token-major` is implemented.
- [x] Existing HTTP mode still works.
- [x] Direct no-profiler run passes on V100.
- [x] Direct windowed profiler run passes on V100 or records a clear profiler
      limitation.
- [x] Artifacts are copied into `logs/from-cluster/`.
- [x] `docs/sprints/STATUS.md` and `TEMP_STATUS_REPORT_059.md` are updated.
- [x] Sprint artifacts are committed.

## Expected Decision

If direct windowed profiling emits real kernel data, use it as the contract for
the first fusion sprint. If it still does not, stop trying profiler API windows
on this stack and use broad direct `nvprof` plus manual trace-window filtering.

## Outcome

Implemented `--run-mode direct-token-major` in the permanent TP/EP profiler
harness. The new mode invokes `tools/ds4-v100-tp-ep-full-layer-smoke`
directly, keeps the same 32-slot / 256K typed-KV serving flags as the HTTP
path, and writes command, stdout/stderr, summary JSON, and parsed
`top-kernels.tsv` into the profiler artifact tree.

V100 direct no-profiler validation passed:

```text
tool: none
slots: 32
ctx: 262144
generated tokens: 64
continuation tokens: 32
generated tok/s decode: 83.882587
continuation tok/s decode: 91.958152
output_head_finite_bad: 0
```

V100 direct windowed `nvprof` validation passed and emitted usable kernel rows:

```text
tool: nvprof-window-gpu-trace
generated tok/s decode: 69.956832
continuation tok/s decode: 82.025033
profiler_marker_lines: 2
output_head_finite_bad: 0
```

Existing HTTP mode was also sanity-checked after the harness change:

```text
tool: none
run mode: http
requests: 1
tokens: 1
http_200: 1
server_generated_tok_s_decode: 91.043258
```

The top windowed kernels were:

```text
TurboMind SM70 FP4 HMMA GEMM: 46.028956 ms, 172 calls
CUTLASS WMMA FP16 GEMM:       14.949741 ms, 720 calls
fill_dense_input_half:        14.745042 ms, 128 calls
compressor_store_slots:       12.823440 ms, 124 calls
bf16_dense_kernel:             7.437483 ms, 1 call
```

A broad direct `nvprof-gpu-trace` run also produced usable kernel rows. Its top
entries confirmed the same broad shape, with `bf16_dense_kernel`,
`f8_b128_to_half_kernel`, CUTLASS WMMA, compressor store, TurboMind FP4 HMMA,
dense-fill, gather, and cast kernels all present in the fixed decode window.

Artifacts are in:

```text
logs/from-cluster/sprint347-direct-tp-ep-profile/cluster/
logs/from-cluster/sprint347-http-sanity/cluster/
```

## Decision

Use direct non-server profiling as the permanent CUDA kernel evidence path for
the TP/EP appliance. HTTP profiler windows remain useful for correctness and
server overhead checks, but direct profiling is now the reliable kernel-level
tool on this V100 stack.

The next optimization sprint should target the measured hot path rather than
another profiler wrapper: `sum_hc_current_input_ms` dominates the direct
no-profiler run (`622.442653 ms` of `762.971220 ms` summed decode). The kernel
trace shows the actionable sub-cost is transform/staging fragmentation around
current-hidden/HC input preparation: dense input fill, BF16/F8 unpack,
gather/cast, and many small WMMA launches.
