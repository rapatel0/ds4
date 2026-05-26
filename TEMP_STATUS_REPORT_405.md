# TEMP_STATUS_REPORT_405

Date: 2026-05-26

## Current Focus

Sprint 405 tested the first half of the NCCL memory plan: make the diagnostic
output head lazy so it is not resident during startup and the 43-layer decode
loop.

## Implementation

- Added `--diagnostic-output-head-lazy-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added profile flag `--lazy-output-head`.
- Added launcher env `DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY`.
- Default behavior is unchanged.
- HTTP serving still uses the resident output-head path; this sprint only
  changes direct serving-bench diagnostics.

## V100 Results

Artifacts:

- `logs/from-cluster/sprint405-lazy-output-head/lazy-control/`
- `logs/from-cluster/sprint405-lazy-output-head/lazy-hc-nccl/`

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | Finding |
|---|---:|---:|---:|---:|---:|---|
| lazy control | 0 | 54639 | 97.034724 | 105.686032 | 68 MiB | Correct, but lazy head leaves GPU0 nearly full |
| lazy + HC-current NCCL | 2 | n/a | n/a | n/a | 1248 MiB | CUDA OOM before first token at compressed KV state allocation |

Important checkpoints:

- Lazy control `after_hc_controls`: `1880 MiB` free.
- Lazy control `after_lazy_output_head`: `68 MiB` free on GPU0.
- Lazy + NCCL `after_hc_controls`: `1248 MiB` free.
- Lazy + NCCL failure: `tools/ds4-v100-tp-ep-full-layer-smoke.cu:9869`,
  compressed KV state allocation on layer 5.

## Decision

Lazy output-head is useful diagnostic infrastructure, but it is not the
production NCCL fix. It moves output-head residency later and preserves first
token in the non-NCCL direct run, but total peak memory is still too high.

Next work should stay on NCCL memory admission and reduce the resident
allocation set before or during decode: GPU0 HC controls, compressed-KV
transients, and then a broader shared NCCL boundary. No PP/layer-split work.
