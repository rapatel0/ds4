# TEMP Status Report 400

Date: 2026-05-26

## Focus

Sprint 400 tested a serving-facing NCCL allgather gate for the TP attention
output boundary. This is TP/EP-only work; no PP/layer-split variants were
touched.

## What Changed

- Added `--true-ds4-attention-output-nccl-allgather-gate`.
- Added launcher/profile wiring via
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER`.
- Reused the existing per-rank NCCL communicator lifecycle when the new gate
  is active.
- Replaced the attention-output A peer-copy gather with NCCL allgather in the
  gated path.
- Added a rank-major-to-slot-major fill kernel so NCCL output can feed the
  second attention output projection.

## V100 Results

Build passed:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Target `32` slot / `256K` direct A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | Attention output ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| peer-copy control | 0 | 45178 | 36.053783 | 39.095421 | 1100.845592 | 1775.125767 | 1746 MiB |
| NCCL allgather | 2 | n/a | n/a | n/a | partial | partial | 1114 MiB |

The candidate failed at the production target shape with:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9719: out of memory
```

The failure is a memory-admission issue, not an immediate semantic failure.
The NCCL candidate emitted many per-layer PASS rows before OOM. The NCCL
communicator adds roughly `+660 MiB/GPU`: rank-buffer max used rose from
`2317` to `2979` MiB, and dense-op max used rose from `30241` to `30901` MiB.

Functional `16` slot / `256K` diagnostic:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | Attention output ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| peer-copy control | 0 | 45178 | 29.467687 | 32.482707 | 594.220063 | 1085.935263 | 4454 MiB |
| NCCL allgather | 0 | 45178 | 28.925690 | 32.202492 | 571.912503 | 1106.283031 | 3820 MiB |

The 16-slot run proves the implementation is functionally correct: the first
token matches. It is not a throughput win: local attention-output timing
improves, but total decode and wall time are slightly worse.

## Decision

Do not promote. Keep the gate diagnostic-only.

NCCL is still promising for true TP hidden/expert collectives, as Sprint 399
showed in the layer-boundary proxy, but this narrow attention-output allgather
does not justify the communicator memory tax at the target `32` slot / `256K`
shape.

## Artifacts

- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-control/`
- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-candidate/`
- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-control-16/`
- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-candidate-16/`

