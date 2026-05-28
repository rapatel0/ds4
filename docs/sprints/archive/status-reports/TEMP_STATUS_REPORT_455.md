# TEMP Status Report 455

## Current Topline

Cleanest current TP/EP serving baseline:

```text
shape:    32 requests / 32 slots / 256K context / 32 generated tokens
artifact: /localpool/ds4/workspace/logs/s455-router-ffn-rankmajor-s32-t32-scratch1280
default:  router+FFN rank-major + fixed-capacity route plan + scratch 1280 MiB
```

Result:

```text
readiness:       pass/pass
response parity: 32/32
server decode:   33.170805 -> 35.578211 tok/s
continuation:    33.156600 -> 35.585793 tok/s
client tok/s:    13.525258 -> 14.801409
avg GPU util:    10.24% -> 11.77%
min free VRAM:   1584 -> 1734 MiB
VRAM failures:   0 -> 0
```

## Implemented This Pass

- Added `DS4_V100_CUDA_LIB_DIR=auto` so the launcher finds the node-local CUDA
  runtime under `/localpool/ds4/cuda-12.2-link/lib64`.
- Promoted `DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB=1280` for the appliance
  launcher/env defaults.
- Recorded Sprint 454 and Sprint 455 artifacts.

## Interpretation

The router+FFN rank-major bundle is a real serving win over the longer decode
window, but it is still not the large utilization unlock. Average GPU
utilization remains around `12%`, with GPU0 still hotter than the other ranks.
Next work should focus on graph-safe launch reduction and HC/post-attention
staging, not PP variants.
