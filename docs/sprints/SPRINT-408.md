# Sprint 408: Post-Close Lazy Output-Head VRAM Checkpoint

Date: 2026-05-26

## Overview

Sprint 407 made HTTP lazy output-head serving work at the target `32` slot /
`256K` shape. The remaining NCCL blocker is reserve admission:
`nccl_after_lazy_output_head` has only `386 MiB` free against the `1536 MiB`
threshold.

Before changing the output-head kernel or format, split the measurement. The
`after_lazy_output_head` checkpoint is taken immediately after opening the
temporary output head and before it is closed. It therefore mixes:

- decode-state residency accumulated during the 43-layer pass;
- temporary output-head weights/logits/workspace;
- CUDA allocator effects.

This sprint adds a checkpoint after the lazy output head is closed, so we can
separate transient output-head peak from post-token steady-state residency.

## Constraints

- TP/EP only. No PP/layer-split work.
- No serving semantics change.
- Default-off lazy behavior stays as implemented in Sprint 407.
- Keep existing `after_lazy_output_head` checkpoint for peak visibility.

## Implementation

File:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

Change:

- After a lazy output-head run succeeds, close the temporary output head and
  emit:
  - `after_lazy_output_head_close`
  - `nccl_after_lazy_output_head_close` when NCCL gates and
    `--nccl-min-free-mib` are active.

## Validation

Local:

```text
git diff --check
```

V100:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run:

1. direct HC-current NCCL + lazy output-head, `32` slots / `256K`
2. HTTP HC-current NCCL + lazy output-head, `32` requests / `32` slots /
   `256K`

Record:

- first token / HTTP status
- `after_lazy_output_head`
- `after_lazy_output_head_close`
- NCCL reserve failure counts before and after close

Artifacts:

- `logs/from-cluster/sprint408-post-close-output-head/direct-lazy-hc-nccl/`
- `logs/from-cluster/sprint408-post-close-output-head/http-lazy-hc-nccl/`

Results:

| Case | Result | First token | Decode tok/s | Continuation tok/s | Before close min free | After close min free | NCCL reserve |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct lazy + HC-current NCCL | returncode 0 | 54639 | 96.275816 | n/a | 386 MiB | 522 MiB | fails 1536 MiB on 8/8 GPUs |
| HTTP lazy + HC-current NCCL | 32/32 HTTP 200 | 83480 | 112.666647 | 110.891026 | 386 MiB | 520 MiB | fails 1536 MiB on 8/8 GPUs |

HTTP response 0 returned generated token sequence `[83480, 79768]`, matching
Sprint 407's lazy HC-current NCCL sequence.

Closing the temporary lazy output head recovers only about `134-136 MiB` on the
tightest GPU. That is useful telemetry, but it does not change NCCL admission:
post-close free VRAM is still roughly `1.0 GiB` below the `1536 MiB` reserve.
The persistent decode-state footprint, not temporary output-head residency, is
now the dominant reserve blocker.

## Definition of Done

- Post-close checkpoint is emitted only for lazy output-head runs.
- V100 build passes.
- Direct target-shape NCCL run records post-close VRAM.
- HTTP target-shape NCCL run records post-close VRAM.
- Sprint doc, status, vision, temporary report, and artifacts are committed.

## Decision Gate

If post-close VRAM passes the `1536 MiB` reserve, update the admission policy
so the output-head peak is treated separately from steady-state NCCL decode
residency. If post-close VRAM still fails, use the deficit to choose the next
real memory reclaim target.

## Decision

Keep the post-close checkpoint as permanent diagnostic telemetry.

Do not relax or move the NCCL admission policy. The target `32` slot / `256K`
HC-current NCCL path serves correctly, but after the lazy output head closes it
still has only `520-522 MiB` free on the tightest GPU, so it remains
diagnostic-only. The next NCCL sprint should reclaim persistent TP/EP decode
state and GPU0-heavy controls rather than spending more time on temporary
output-head close timing.
