# TEMP Status Report 058 - TP/EP Profiler Window

Date: 2026-05-25

## What Changed

Sprint 346 added decode-window profiler support to the TP/EP path:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu` now accepts
  `--cuda-profiler-window`.
- `tools/ds4-v100-run-appliance.sh` now forwards
  `DS4_V100_CUDA_PROFILER_WINDOW=1` in TP/EP mode.
- `tools/ds4-v100-tp-ep-profile.py` has permanent windowed profiler modes:
  `nvprof-window-gpu-trace`, `nvprof-window-api-trace`, `ncu-window-basic`,
  and `ncu-window-nvlink`.

Normal serving has no profiler API calls unless the flag/env is enabled.

## V100 Validation

Build:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

No-profiler sanity:

```text
32/32 HTTP 200
32 slots
256K context
2 tokens/request
coalesced_batch_size = 32
server tok/s = 78.032873
decode tok/s = 92.070787
```

Windowed `nvprof`:

```text
32/32 HTTP 200
36 profiler start/stop marker lines
server tok/s = 72.078755
decode tok/s = 84.920155
trace file size = 0 bytes
```

Windowed `ncu`:

```text
32/32 HTTP 200
34 profiler start/stop marker lines
server tok/s = 35.193444
decode tok/s = 39.720168
ncu report = process lifecycle lines only
```

## Interpretation

The profiler window implementation is wired and observable, but the current
HTTP-wrapper profiler setup still does not produce scoped kernel metrics.
`nvprof --profile-from-start off` writes an empty trace, and Nsight Compute
connects but does not emit kernel metrics before the harness terminates the
server.

Sprint 345's broad `nvprof` trace remains the current actionable performance
evidence:

```text
cutlass WMMA/HMMA         43496 calls   544.815 ms
compressor kernels        10600 calls   434.288 ms
gather kernels            46040 calls   429.057 ms
fill_dense kernels        46032 calls   422.979 ms
TurboMind SM70 FP4 HMMA    1300 calls   348.017 ms
```

## Next Step

Build a direct non-server TP/EP profile target that reuses the resident
32-slot typed decode path, runs one or two decode windows, and exits naturally.
That should give NCU/nvprof a clean process lifetime and avoid profiling the
HTTP server lifecycle. Once that target produces clean metrics, fuse the
largest non-GEMM boundary first: HC-current gather/fill, dense fill/gather, or
compressor staging depending on the direct profile ranking.
