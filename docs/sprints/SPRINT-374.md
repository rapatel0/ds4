# Sprint 374: V100 INT8 Compressor Workbench

## Overview

Build and run the first focused V100 kernel workbench for the measured TP/EP
compressed-KV dense bottleneck.

Sprint 373 showed that the best scoped INT8 candidate is not a whole-model
conversion. It is the BF16 attention compressor family currently executed at:

```text
M = 32 slots
N = 128 or 64 TP-shard rows
K = 4096 hidden input
```

The goal of this sprint is to answer whether our copied V100 INT8 kernels are
worth wiring into the production TP/EP compressor path.

## Scope

- Add a permanent CUDA workbench for the two BF16 attention-compressor shapes.
- Compare:
  - cuBLAS FP16 tensor-core baseline as the current BF16-on-V100 execution
    proxy.
  - tc-grid `v12s_64x128x32_w4_ks8`, the dispatcher-preferred small-M kernel.
  - tc-grid `v13_rf_v6_128x128x16_w4`, the larger-M champion, as a shape
    sanity check.
- Validate numerical correctness against a CPU reference for the INT8 path and
  report quantization error.
- Emit machine-readable TSV and a markdown summary.
- Run on the V100 pod and commit artifacts.

## Out Of Scope

- Do not change production TP/EP runtime math in this sprint.
- Do not convert the real pack yet.
- Do not promote INT8 by default.
- Do not restart PP/layer-split work.
- Do not vendor the full vLLM TurboMind stack in this sprint.

## Reference Inputs

- Current bottleneck map:
  `docs/architecture/DS4-V100-TP-EP-BOTTLENECKS.md`
- Candidate audit:
  `docs/sprints/SPRINT-373.md`
- Existing copied tc-grid kernels:
  `kernels/tc-grid/`
- V100 vLLM/TurboMind fork for shape/config ideas:
  `research/1Cat-vLLM/lmdeploy/src/turbomind/kernels/gemm/`

The vLLM fork confirms that its SM70 TurboMind registry also treats `M=32`
as a first-class decode shape, with small `CTA_M=32`, `CTA_N=128`, and
split/tuning-heavy dispatch for V100. That makes the current workbench shape
credible and gives a follow-up path if tc-grid is not enough.

## Implementation

Add:

```text
tools/ds4-v100-tp-ep-int8-compressor-workbench.cu
tools/ds4-v100-tp-ep-int8-compressor-workbench
```

The tool should support:

```text
--iters N
--warmup N
--out-tsv PATH
--report PATH
```

It should run both target shapes:

```text
M=32,N=128,K=4096
M=32,N=64,K=4096
```

For each shape, record:

- kernel label
- mean milliseconds
- effective dense TFLOP/s
- effective memory GB/s
- max/p99/mean absolute error versus CPU reference where applicable
- whether the kernel passed correctness

## Definition Of Done

- Workbench builds on the V100 pod with `CUDA_ARCH=sm_70`.
- Workbench runs both compressor shapes.
- Output artifacts are copied to
  `logs/from-cluster/sprint374-int8-compressor-workbench`.
- Sprint doc records the measured result and decision.
- `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- Changes are committed.

## Expected Decision

If tc-grid INT8 materially beats the FP16 baseline for `M=32,N=128/64,K=4096`
with acceptable numerical error, the next sprint should add an offline pack
variant and a gated runtime path for only `attn_compress_{kv,gate}.weight`.

If it is flat or slower, do not wire it into production. The next kernel path
should either adapt the vLLM TurboMind SM70 small-M GEMM registry/dispatcher or
move to a fused compressor/state kernel that changes the execution boundary
instead of just swapping GEMM dtype.

## Implementation Result

Added:

```text
tools/ds4-v100-tp-ep-int8-compressor-workbench.cu
tools/ds4-v100-tp-ep-int8-compressor-workbench
```

The workbench builds a deterministic compressor-shaped problem, quantizes
weights to the tc-grid `int8 + fp16 scale per row/per-32K` layout, and compares:

- cuBLAS FP16 tensor-op baseline with FP32 output.
- tc-grid `v12s_64x128x32_w4_ks8` with required split-K output zeroing.
- tc-grid `v13_rf_v6_128x128x16_w4`.

The vLLM/TurboMind fork was reviewed for SM70 shape guidance. Its registry has
explicit small-M V100 kernels including `CTA_M=32`, `CTA_N=128`, and broader
V100 tuning around split-K/waves. That supports treating these compressor
shapes as first-class decode shapes, but this sprint did not vendor or wire the
larger TurboMind stack.

## V100 Result

Command:

```text
CUDA_VISIBLE_DEVICES=0 ./tools/ds4-v100-tp-ep-int8-compressor-workbench \
  --warmup 30 \
  --iters 300 \
  --out-tsv /workspace/logs/sprint374-int8-compressor-workbench/int8-compressor-workbench.tsv \
  --report /workspace/logs/sprint374-int8-compressor-workbench/INT8_COMPRESSOR_WORKBENCH.md
```

Topline:

| M | N | K | Kernel | ms | TFLOP/s | GB/s | max abs | p99 abs | OK |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 4096 | `cublas-f16-tensorop` | 0.009250 | 3.627 | 143.469 | 0.000109 | 0.000088 | 1 |
| 32 | 128 | 4096 | `tc-grid-v12s-ks8+zero` | 0.042721 | 0.785 | 25.695 | 0.015222 | 0.012570 | 1 |
| 32 | 128 | 4096 | `tc-grid-v13-rf-v6` | 0.282150 | 0.119 | 3.891 | 0.013401 | 0.011393 | 1 |
| 32 | 64 | 4096 | `cublas-f16-tensorop` | 0.008803 | 1.906 | 90.268 | 0.000109 | 0.000089 | 1 |
| 32 | 64 | 4096 | `tc-grid-v12s-ks8+zero` | 0.036673 | 0.457 | 22.115 | 0.015222 | 0.012570 | 1 |
| 32 | 64 | 4096 | `tc-grid-v13-rf-v6` | 0.260751 | 0.064 | 3.110 | 0.013401 | 0.011393 | 1 |

The INT8 kernels are numerically acceptable for this synthetic quantized
problem, but they are not performance candidates for the current compressor
GEMMs:

- `N=128`: best tc-grid INT8 is `4.62x` slower than cuBLAS FP16.
- `N=64`: best tc-grid INT8 is `4.17x` slower than cuBLAS FP16.
- `v13` is the wrong shape regime for `M=32`.

## Decision

Do not wire the copied tc-grid INT8 kernels into the production
`attn_compress_{kv,gate}.weight` path.

The next optimization should not be a simple BF16-to-INT8 GEMM swap. The two
credible follow-ups are:

1. Adapt the vLLM/TurboMind SM70 small-M GEMM path for this exact compressor
   shape and compare it to cuBLAS FP16.
2. Fuse the compressor dense boundary with adjacent state/emit work so the
   runtime removes launches, staging, and format traffic rather than only
   changing the dense kernel dtype.

## Validation

V100:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-int8-compressor-workbench
```

Artifacts:

- Cluster: `/workspace/logs/sprint374-int8-compressor-workbench`
- Local: `logs/from-cluster/sprint374-int8-compressor-workbench`
