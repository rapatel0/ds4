# TEMP Status Report 059 - Direct TP/EP Profiling

Date: 2026-05-25

## Current Focus

TP/EP-only serving path. PP/layer-split work remains frozen. MTP is still
deferred until TP/EP correctness and serving performance are operational.

## What Changed

Sprint 347 added `--run-mode direct-token-major` to the permanent profiler
harness:

```text
tools/ds4-v100-tp-ep-profile.py --run-mode direct-token-major
```

This bypasses the HTTP wrapper and runs
`tools/ds4-v100-tp-ep-full-layer-smoke` directly with the same 32-slot / 256K
typed-KV serving flags. It records command lines, stdout/stderr, summary JSON,
and parsed `top-kernels.tsv` files.

## V100 Results

Direct no-profiler:

```text
slots: 32
ctx: 262144
decode steps: 2
generated tokens: 64
continuation tokens: 32
generated tok/s decode: 83.882587
continuation tok/s decode: 91.958152
output_head_finite_bad: 0
```

Direct windowed nvprof:

```text
generated tok/s decode: 69.956832
continuation tok/s decode: 82.025033
profiler_marker_lines: 2
output_head_finite_bad: 0
```

HTTP mode sanity after the harness change:

```text
requests: 1
tokens: 1
http_200: 1
server_generated_tok_s_decode: 91.043258
```

Top windowed kernels:

```text
TurboMind SM70 FP4 HMMA GEMM: 46.028956 ms, 172 calls
CUTLASS WMMA FP16 GEMM:       14.949741 ms, 720 calls
fill_dense_input_half:        14.745042 ms, 128 calls
compressor_store_slots:       12.823440 ms, 124 calls
bf16_dense_kernel:             7.437483 ms, 1 call
```

Broad direct nvprof also produced kernel rows, confirming that BF16/F8 unpack,
gather, dense-fill, cast, compressor, CUTLASS, and TurboMind kernels are active
in the fixed decode window.

## Interpretation

The profiler evidence does not support "tensor cores are not being used."
TurboMind FP4 HMMA and CUTLASS WMMA kernels are present. The issue is that the
serving path is still fragmented around transform/staging boundaries.

The most important direct timer is:

```text
sum_hc_current_input_ms = 622.442653
sum_decode_ms          = 762.971220
```

So the next optimization should target HC/current-input staging and the related
gather/cast/fill/unpack chain, not another profiler wrapper.

## Artifacts

```text
logs/from-cluster/sprint347-direct-tp-ep-profile/cluster/
logs/from-cluster/sprint347-http-sanity/cluster/
```

Raw profiler CSVs were intentionally not committed. The committed artifacts
keep command lines, summaries, stdout/stderr, and parsed top-kernel tables.
