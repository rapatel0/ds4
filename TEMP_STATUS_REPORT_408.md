# TEMP Status Report 408

Date: 2026-05-26

## Current Focus

Only TP/EP work is active. PP/layer-split variants are frozen as historical
baselines. The current workstream is NCCL admission for the target
`32` slot / `256K` serving shape.

## Sprint 408 Result

Implemented a post-close lazy output-head VRAM checkpoint in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

New checkpoint labels:

- `after_lazy_output_head_close`
- `nccl_after_lazy_output_head_close`

The existing peak checkpoints remain unchanged.

## V100 Evidence

| Case | Result | First token | Decode tok/s | Continuation tok/s | Before close min free | After close min free |
|---|---:|---:|---:|---:|---:|---:|
| Direct HC-current NCCL + lazy output-head | returncode 0 | 54639 | 96.275816 | n/a | 386 MiB | 522 MiB |
| HTTP HC-current NCCL + lazy output-head | 32/32 HTTP 200 | 83480 | 112.666647 | 110.891026 | 386 MiB | 520 MiB |

HTTP response 0 generated token sequence `[83480, 79768]`.

## Decision

Closing the lazy output head recovers only `134-136 MiB`. The post-close
steady-state still has only `520-522 MiB` free on the tightest GPU, far below
the `1536 MiB` NCCL reserve. HC-current NCCL remains diagnostic-only.

The next NCCL sprint should target persistent TP/EP decode-state residency and
GPU0-heavy control buffers. Output-head timing alone is not enough.

## Artifacts

- `logs/from-cluster/sprint408-post-close-output-head/direct-lazy-hc-nccl/`
- `logs/from-cluster/sprint408-post-close-output-head/http-lazy-hc-nccl/`
