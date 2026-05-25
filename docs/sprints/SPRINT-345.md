---
sprint: 345
title: TP/EP Serving Profiler Harness
status: paused-for-review
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 345 - TP/EP Serving Profiler Harness

## Goal

Make TP/EP serving profiling repeatable and collect the first profiler-backed
evidence for the low-utilization 32-slot typed-KV path.

## Scope

TP/EP only. No PP/layer-split work. No MTP. Keep profiling fully opt-in so the
production appliance path has zero runtime overhead unless the diagnostic
harness is explicitly launched.

## Outcome

Added `tools/ds4-v100-tp-ep-profile.py`, a permanent diagnostic harness that
starts the TP/EP appliance server, drives concurrent HTTP requests, and can
wrap the server with:

- `--tool none`
- `--tool nvprof-gpu-trace`
- `--tool nvprof-api-trace`
- `--tool ncu-basic`
- `--tool ncu-nvlink`

The Nsight Compute modes support `--ncu-launch-count`, `--ncu-launch-skip`,
and `--ncu-kernel-name` filters so future profiling can target a small set of
kernels instead of instrumenting the full serving launch stream.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
```

Cluster:

```text
pod: llm/llamacpp-build-8gpu
repo: /workspace/ds4-sprint181
ctx: 262144
slots: 32
requests: 32 concurrent
typed KV: history + skip-current-load + quiet + batch-rows + stream-sync
```

No-profiler sanity:

```text
http_200: 32/32
coalesced_batch_size: 32
tokens/request: 2
generated_tokens_meta: 64
server_generated_tok_s: 79.571869
server_generated_tok_s_decode: 94.059803
```

`nvprof --print-gpu-trace`:

```text
http_200: 32/32
coalesced_batch_size: 32
tokens/request: 2
generated_tokens_meta: 64
server_generated_tok_s: 62.986544
server_generated_tok_s_decode: 73.352688
kernel_launches: ~274495
memcpy_events: ~177864
```

Top kernel families in the parsed trace:

```text
family                    calls   total_ms
cutlass WMMA/HMMA         43496   544.815258
compressor kernels        10600   434.288469
gather kernels            46040   429.056773
fill_dense kernels        46032   422.979145
TurboMind SM70 FP4 HMMA    1300   348.017383
cublasLt splitKreduce     38296   117.189575
typed KV store             6240    44.717453
typed KV load              1008    28.144692
```

The raw trace confirms that tensor-core-capable kernels are present:

- `cutlass_70_wmma_tensorop_s161616gemm_f16_32x32_64x2_tn_align8`
- TurboMind `Sm70` kernel using `SM70_MMA_884` with packed FP4 operand B

However, the trace also confirms severe launch fragmentation. The serving
window is not dominated by the typed KV row load/store kernels themselves;
they are visible but much smaller than dense fill/gather/compressor work and
the massive number of small WMMA/cuBLAS launches.

Nsight Compute attempt:

```text
tool: ncu-basic
kernel filter: regex:.*cutlass_70_wmma.*
launch count: 8
http_200: 32/32
tokens/request: 1
server_generated_tok_s_decode: 50.929094
```

This run did not emit useful kernel metrics. The report only contains Nsight
Compute process lifecycle lines, ending with `Received signal`, because the
server harness terminates the appliance after the HTTP requests finish. Future
NCU measurement should use either a CUDA profiler start/stop window inside the
binary or a non-server replay target that exits naturally after the profiled
kernel window.

## Decision

Pause here for review. The actionable finding is not "we are missing tensor
cores"; both Cutlass WMMA/HMMA and TurboMind SM70 HMMA kernels are present.
The stronger finding is that the TP/EP prototype is launch- and transform-heavy:
hundreds of thousands of small launches for a tiny 64-token test, with
fill/gather/compressor kernels comparable to or larger than the expert GEMM
time. The next optimization discussion should focus on collapsing the
HC-current, dense-input fill, gather, compressor, and GEMM boundaries before
spending more time on typed KV row micro-optimizations.

## Artifacts

- `logs/from-cluster/sprint345-nsight-typed-serving/cluster/none/summary.json`
- `logs/from-cluster/sprint345-nsight-typed-serving/cluster/nvprof-gpu-trace/top-kernels.tsv`
- `logs/from-cluster/sprint345-nsight-typed-serving/cluster/nvprof-gpu-trace/summary.json`
- `logs/from-cluster/sprint345-nsight-typed-serving/cluster/ncu-basic/ncu-basic.csv`
- `logs/from-cluster/sprint345-nsight-typed-serving/cluster/ncu-basic/summary.json`
