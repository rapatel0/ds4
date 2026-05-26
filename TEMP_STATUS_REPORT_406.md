# TEMP_STATUS_REPORT_406

Date: 2026-05-26

## Current Focus

Sprint 406 continued the TP/EP NCCL memory work. The goal was to fix the
compressed-KV state allocation that caused Sprint 405's target-shape NCCL run
to OOM at layer 5.

## Implementation

Changed `tools/ds4-v100-tp-ep-full-layer-smoke.cu` so attention compressed-KV
state uses exact DS4 per-ratio dimensions:

| Ratio | Old state layout | New state layout |
|---:|---:|---:|
| 4 | `128 x 1024` | `8 x 1024` |
| 128 | `128 x 1024` | `128 x 512` |

This is not a dtype change and does not quantize anything. It removes padded
resident state that the kernels did not semantically need.

## V100 Results

Artifacts:

- `logs/from-cluster/sprint406-compact-kv-state/lazy-control/`
- `logs/from-cluster/sprint406-compact-kv-state/lazy-hc-nccl/`

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | Result |
|---|---:|---:|---:|---:|---:|---|
| lazy control | 0 | 54639 | 78.447208 | 77.220402 | 1018 MiB | Correct; peak free improved materially |
| lazy + HC-current NCCL | 0 | 54639 | 89.952595 | 100.096637 | 386 MiB | Correct first-token completion; reserve still fails |

Sprint 405 comparison:

- Non-NCCL `after_lazy_output_head` improved from `68 MiB` free to `1018 MiB`.
- HC-current NCCL moved from CUDA OOM at layer 5 to successful first-token
  completion at `32` slots / `256K`.

## Decision

Promote the compact compressed-KV state layout.

Keep HC-current NCCL diagnostic-only for now. It completes at the target
shape, but `nccl_after_lazy_output_head` fails the `1536 MiB` reserve on all
eight GPUs, with GPU0 at `386 MiB` free.

## Next

The next sprint should keep focus on NCCL production admission:

- make lazy/on-demand output-head compatible with the HTTP serving path;
- reduce remaining peak VRAM so NCCL clears the `1536 MiB` reserve;
- then rerun target-shape HTTP readiness/parity before promoting NCCL.
