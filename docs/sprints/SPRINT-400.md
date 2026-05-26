# Sprint 400: NCCL Attention Output AllGather Gate

Date: 2026-05-26

## Overview

Stay on TP/EP only and attach the Sprint 399 NCCL result to a serving-facing
TP boundary. The current true DS4 attention-output path computes
`attn_output_a` shards on all ranks, then every rank rebuilds the full
`attn_output_a` activation with `cudaMemcpy2DAsync` from every other rank
before feeding the second attention output projection.

This sprint adds a default-off NCCL all-gather path for that boundary. Because
NCCL all-gather emits rank-major shard order while the downstream dense input
expects slot-major rows, the implementation must include an explicit
rank-major-to-slot-major fill kernel.

## Constraints

- TP/EP only. No PP/layer-split work.
- No generic scheduler abstraction.
- Default off until V100 A/B proves it.
- Preserve first-token evidence and existing attention-output semantics.
- Keep `32` slots and `256K` context as the measurement shape.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`

Planned changes:

1. Add `--true-ds4-attention-output-nccl-allgather-gate`.
2. Add launcher/profile wiring:
   `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER`.
3. Reuse the existing per-rank NCCL communicator lifecycle when the new gate is
   active.
4. Replace the attention-output A peer-copy gather with NCCL all-gather in the
   compatible path.
5. Convert NCCL rank-major gathered output into slot-major half dense input for
   the second projection.
6. Leave the existing peer-copy path as fallback.

## Validation

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS on the V100 pod.

Direct same-binary A/B:

- `32` slots
- `256K` context / `position=262080`
- model-router compact-MoE
- route-plan async upload enabled
- true DS4 attention output gate enabled
- candidate adds only the NCCL all-gather gate

Record:

- first token
- generated/continuation decode tok/s
- `sum_pre_ep_attention_output_ms`
- total decode ms
- VRAM admission

## Results

Target `32` slot / `256K` direct A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | Attention output ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| peer-copy control | 0 | 45178 | 36.053783 | 39.095421 | 1100.845592 | 1775.125767 | 1746 MiB |
| NCCL allgather | 2 | n/a | n/a | n/a | partial | partial | 1114 MiB |

The NCCL candidate ran through many layers with per-layer
`tp_ep_true_attention_output_projection ... nccl_allgather 1 ... PASS` rows,
then failed with CUDA OOM while allocating the next layer's raw-SWA state:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9719: out of memory
```

The measured memory delta is consistent with NCCL communicator overhead:
`vram_after_rank_buffers_max_used_mib` rose from `2317` to `2979`, and
`vram_after_dense_ops_max_used_mib` rose from `30241` to `30901`, roughly
`+660 MiB/GPU`. That erases the target-shape headroom.

Functional diagnostic `16` slot / `256K` A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | Attention output ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| peer-copy control | 0 | 45178 | 29.467687 | 32.482707 | 594.220063 | 1085.935263 | 4454 MiB |
| NCCL allgather | 0 | 45178 | 28.925690 | 32.202492 | 571.912503 | 1106.283031 | 3820 MiB |

The `16` slot run proves the rank-major NCCL allgather plus slot-major fill is
functionally correct, but it is not a topline win: attention-output timing
improves locally while total decode and wall time regress slightly.

Artifacts:

- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-control/`
- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-candidate/`
- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-control-16/`
- `logs/from-cluster/sprint400-attn-output-nccl-allgather/direct-candidate-16/`

## Decision

REJECT as a default and keep diagnostic-only.

NCCL remains strategically useful for true TP hidden/expert collectives, but
this specific attention-output allgather is not promotable at the production
`32` slot / `256K` shape. It adds about `0.6-0.7 GiB/GPU` of communicator
resident memory, fails the target run by OOM, and is only flat/slightly slower
at the smaller `16` slot shape.

Next NCCL work should either:

1. amortize one shared communicator across multiple necessary TP collectives
   with explicit memory admission, or
2. target a larger fused TP boundary where the collective replaces more than a
   narrow peer-copy gather.

## Definition of Done

- Gate exists in binary, launcher, and profile harness.
- V100 build passes.
- Direct A/B records correctness evidence and attention-output timing.
- Sprint doc, temporary status report, status, and vision are updated with an
  explicit PROMOTE or REJECT decision.
- Commit all kept artifacts explicitly.

## Risks

- NCCL all-gather may be faster but the layout conversion kernel may erase the
  win at 32 slots.
- The true attention-output path may not be enabled in the current default
  serving profile, so this is a serving-facing diagnostic rather than a default
  serving change.
- Reusing the existing NCCL communicator field keeps the sprint small, but the
  next integration may deserve a renamed general TP/EP communicator.
