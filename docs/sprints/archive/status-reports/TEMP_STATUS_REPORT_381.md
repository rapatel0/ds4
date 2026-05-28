# TEMP Status Report 381

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 381 added and tested a default-off FP8 E5M2 KV-cache gate:

```text
--fp8-e5m2-kv-gate
DS4_V100_TP_EP_FP8_E5M2_KV=1
tools/ds4-v100-tp-ep-profile.py --fp8-e5m2-kv
```

This keeps the existing block-128 layout:

```text
1 E8M0 scale byte + 128 FP8 payload bytes
```

So E5M2 is not a capacity win versus E4M3. It tests numeric format behavior
inside the existing sharded typed-KV runtime.

## What Changed

- Added `DS4_V100_TP_KV_F8_E5M2_B128`.
- Added host/device E5M2 quant/dequant.
- Generalized typed F8 KV store/load kernels to dispatch E4M3 or E5M2.
- Added deterministic E8M0 scale-byte selection from `amax / fp8_max`.
- Added runtime smoke parser support for `--kv-dtype f8_e5m2_b128`.
- Added full-layer, launcher, and profile gates.

## V100 Validation

Artifacts:

```text
/workspace/logs/sprint381-e5m2-kv/
```

Build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-runtime-smoke \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Passed on gpu-01.

Row tests:

| Test | Result |
|---|---|
| E5M2 typed `attn` | `bad_values=0`, `byte_mismatches=0` |
| E5M2 typed `attn_raw` | `bad_values=0`, `byte_mismatches=0` |
| E5M2 typed `indexer` | `bad_values=0`, `byte_mismatches=0` |
| E5M2 device rows | all tested kinds `bad_values=0`, `max_abs=0` |
| E4M3 regression | typed/device `attn` still passed |

## Throughput Evidence

Direct, `32` slots / `256K` / selected-token shape:

| Run | First token | Checksum | Decode tok/s | Continuation tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|
| 1-token E4M3 | 54639 | n/a | 67.710842 | 0 | 107.728597 ms |
| 1-token E5M2 | 54639 | n/a | 69.225694 | 0 | 107.091969 ms |
| 4-token E4M3 | 98751 | 13373834059 | 70.710875 | 75.203353 | 466.656134 ms |
| 4-token E5M2 | 98751 | 13373834059 | 75.787866 | 78.105479 | 413.206374 ms |

HTTP selected-token, `32` requests / `32` slots / `256K` / `4` tokens:

| Run | HTTP 200 | First token | Client tok/s | Avg GPU util | Max GPU util | Max memory |
|---|---:|---:|---:|---:|---:|---:|
| E4M3 control | 32/32 | 45178 | 17.212677 | 1.738636% | 39% | 32418 MiB |
| E5M2 retry | 32/32 | 45178 | 22.389190 | 2.737500% | 39% | 32418 MiB |

One immediate E5M2 HTTP run after the control failed before readiness with CUDA
OOM while allocating the dense cache. Retrying E5M2 alone passed. This means
the default `32` slot / `256K` shape is still operating with very little VRAM
slack.

## Decision

E5M2 KV stays default-off.

It is promising and correctness-clean in the tests above, but it should not
replace E4M3 yet because:

- E5M2 has less mantissa precision than E4M3.
- The proof is short selected-token parity, not a longer chat/quality soak.
- The serving shape is close enough to the VRAM limit that admission/memory
  margin needs hardening.

## Next Best Work

1. Add a longer deterministic parity/soak gate for E5M2 if we want to promote
   it.
2. Add explicit VRAM admission/margin reporting for the `32` slot / `256K`
   TP/EP service.
3. Continue launch/count and staging reductions only after the default serving
   shape has reliable memory slack.
