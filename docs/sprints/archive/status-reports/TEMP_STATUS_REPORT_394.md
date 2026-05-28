# TEMP Status Report 394: Fast Hash-Router Gate

Date: 2026-05-25

## Topline

Sprint 394 implemented and tested `--router-hash-fast-gate`, a default-off
TP/EP model-router optimization.

Result: correct but not promoted. The gate preserves response parity and
readiness, but it does not materially reduce the measured router boundary.

## What Changed

- Added `router_select_hash_fast_rows_kernel` in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added `--router-hash-fast-gate`.
- Added launcher env:
  `DS4_V100_TP_EP_ROUTER_HASH_FAST=1`.
- Added profile harness support:
  `tools/ds4-v100-tp-ep-profile.py --router-hash-fast`.
- Added active-slot matrix suffix support.

The kernel is only used when hash rows, router tokens, and `hash_rows > 0` are
available. Otherwise the existing router select kernel remains the fallback.

## Validation

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Build passed with only existing unused-function warnings.

Same-binary HTTP A/B:

```text
32 requests / 32 slots / 256K ctx / position 262080 / 32 generated tokens
model-router routes / compact MoE / prompt-file soak / VRAM report
```

| Metric | Control | Router hash fast |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| Response parity | `32/32` | `32/32` |
| Readiness | `true` | `true` |
| First token | `83484` | `83484` |
| Server decode tok/s | `106.900859` | `107.274556` |
| Client generated tok/s | `37.231411` | `38.262372` |
| Avg GPU util | `9.296875%` | `9.441489%` |
| Router select ms | `27.766750` | `27.683134` |
| HC-current FFN/router ms | `36.211953` | `36.287395` |
| Scaffold decode ms | `289.821429` | `293.484520` |
| Compressed-KV sum ms | `3285.935154` | `3317.395070` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

Permanent validators:

```text
response parity: match=true, matched_pairs=32, failed_pairs=0
control readiness: ready=true, failure_count=0
candidate readiness: ready=true, failure_count=0
```

## Assessment

The optimization is too small to promote. It proves that evaluating all 256
expert probabilities before applying the DS4 hash row is not the real
performance bottleneck. The remaining router/HC-current cost is broader:
dense logits, host route-planning semantics, synchronization, route upload,
and downstream compose.

## Next

Keep `router-hash-fast` as an opt-in diagnostic. The next sprint should target
a wider boundary, likely fusing/removing host route planning with expert
dispatch/compose or restructuring HC-current/router scheduling so GPU0 is not
serializing the step.
