# TEMP Status Report 414

Date: 2026-05-26

## Current Focus

Sprint 414 closed the immediate semantic TP/EP measurement blocker: diagnostic
stats collection was still running inside the true attention-output and
post-attention FFN-input timed sections. The user steer was to keep focusing
NCCL/semantic TP/EP work and allow fewer than `32` slots when that improves
practical serving.

## What Changed

- Added `DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=auto`.
- Added `--true-ds4-semantic-skip-stats-gate`.
- Launcher auto-enables the gate only for TP/EP semantic serving with
  `DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=1`.
- Profile and HTTP A/B harnesses expose enable/disable switches so future
  runs can still collect diagnostic stats when needed.
- The gate skips host-visible tensor stats in:
  - true attention-output projection
  - post-attention FFN-input materialization
- The gate does not skip semantic kernels, NCCL, router planning, KV, HTTP
  serving, or MoE execution.

## V100 Results

Shape:

```text
ctx      = 262144
position = 262080
requests = 28
slots    = 28
tokens   = 32/request
```

| Case | HTTP | Ready | Server decode tok/s | Client generated tok/s | GPU util avg | GPU util max | Min free VRAM | VRAM failures | Attention output ms | Post-attn ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no skip | `28/28` | `true` | `19.708590` | `7.543522997079557` | `5.844230769230769%` | `24.0%` | `1790 MiB` | `0` | `460.797268` | `129.119597` |
| skip stats | `28/28` | `true` | `31.091919` | `10.366506446092782` | `7.899436090225564%` | `37.0%` | `1790 MiB` | `0` | `19.520681` | `82.034351` |

Response-0 semantic token sequence matched exactly between no-skip and
skip-stats candidates.

## Decision

Promote semantic skip-stats as the production semantic-serving default. This
is a diagnostic synchronization removal, not a model math change.

## Current Topline

- Practical semantic tier remains `28` slots / `256K`.
- Best measured semantic post-attention server decode is now
  `31.091919 tok/s`.
- The promoted fast control at the same `28` slot shape is still about
  `98 tok/s`, so semantic serving remains roughly `3.2x` slower than the
  non-semantic control.
- VRAM is clean at `28` slots with `1790 MiB` minimum free and zero reserve
  failures.

## Remaining Bottleneck

After removing stats synchronization, the real semantic bottleneck is the
post-attention/NCCL boundary, especially the GPU0-centered full-hidden
materialization and broadcast in `run_true_ds4_post_attention_ffn_input`.
The next sprint should stay on NCCL/semantic TP/EP and replace that full-hidden
gather/broadcast with a sharded or collective path before trying to climb back
to `32` slots.

## Artifacts

```text
logs/from-cluster/sprint414-semantic-noskip-28slot-http-ab/
logs/from-cluster/sprint414-semantic-skip-stats-28slot-http-ab/
```
