# TEMP Status Report 057 - TP/EP Profiler Pass

Date: 2026-05-25

## Current State

I added a permanent opt-in profiler harness:

```text
tools/ds4-v100-tp-ep-profile.py
```

It does not add hot-path runtime instrumentation. Default serving remains
unchanged. Profiling only happens when this separate harness wraps the server
with `nvprof` or Nsight Compute.

## Runs Completed

Shape:

```text
32 slots
32 concurrent requests
256K context
typed KV history + skip-current-load + quiet + batch-rows + stream-sync
diagnostic output head
```

No-profiler sanity:

```text
32/32 HTTP 200
coalesced_batch_size = 32
64 generated tokens
server tok/s = 79.571869
decode tok/s = 94.059803
```

`nvprof` GPU trace:

```text
32/32 HTTP 200
coalesced_batch_size = 32
64 generated tokens
server tok/s = 62.986544
decode tok/s = 73.352688
kernel launches ~= 274,495
memcpy events ~= 177,864
```

Nsight Compute filtered attempt:

```text
filter = regex:.*cutlass_70_wmma.*
launch_count = 8
32/32 HTTP 200
32 generated tokens
decode tok/s = 50.929094 under profiling overhead
```

The NCU run did not emit useful per-kernel metrics because the server wrapper
terminates the process after the request batch. We need an internal profiler
window or a non-server replay target for clean NCU reports.

## Top Trace Findings

Top parsed `nvprof` kernel families:

```text
cutlass WMMA/HMMA         43496 calls   544.815 ms
compressor kernels        10600 calls   434.288 ms
gather kernels            46040 calls   429.057 ms
fill_dense kernels        46032 calls   422.979 ms
TurboMind SM70 FP4 HMMA    1300 calls   348.017 ms
cublasLt splitKreduce     38296 calls   117.190 ms
typed KV store             6240 calls    44.717 ms
typed KV load              1008 calls    28.145 ms
```

Interpretation:

- Tensor-core-capable kernels are present.
- TurboMind FP4-on-SM70 kernels are present.
- The typed KV row load/store kernels are not the main remaining cost.
- The path is heavily launch fragmented, with many small transform/fill/gather
  kernels around the GEMMs.

## Current Assessment

The low GPU utilization is more consistent with launch fragmentation and
unfused transform boundaries than with "no tensor cores selected." The next
optimization should collapse data-movement and transform boundaries around
HC-current input, dense fill/gather, compressor emit/store, and GEMM dispatch.

Artifacts are under:

```text
logs/from-cluster/sprint345-nsight-typed-serving/cluster/
```
