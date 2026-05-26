# TEMP Status Report 411

Current focus: TP/EP semantic serving path only. PP/layer-split remains frozen.

## Result

Sprint 411 exposed the existing true-attention output plus post-attention
FFN-input path through the normal HTTP serving harness.

Artifact:

```text
logs/from-cluster/sprint411-true-attn-http-ab-rerun/
```

Shape:

```text
32 concurrent HTTP requests
32 configured slots
262144 context
position 262080
32 generated tokens/request
HC-current NCCL enabled
lazy output head enabled
compact MoE decode enabled
model-router routes enabled
```

## Topline

| Metric | Fast control | Post-attn candidate |
|---|---:|---:|
| HTTP 200 | 32/32 | 32/32 |
| readiness | true | false |
| server generated decode tok/s | 108.084959 | 20.315962 |
| server continuation decode tok/s | 107.964491 | 20.308358 |
| client generated tok/s | 17.810479 | 8.693548 |
| avg sampled GPU util | 4.198034% | 7.124306% |
| min free VRAM | 2106 MiB | 1328 MiB |
| VRAM failures | 0 | 62 |
| attention output timer | 0.0 ms | 512.629430 ms |
| post-attn FFN input timer | 0.0 ms | 144.063057 ms |

Decision:
`true-attention-post-attention-serving-served-reserve-blocked`.

The candidate did not fail decode: it returned all responses and the new
semantic timers were active. It failed readiness because it crossed the current
`1536 MiB` NCCL reserve threshold.

## What Changed

- Added launcher/env support for
  `DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT`.
- Added `tools/ds4-v100-tp-ep-profile.py --post-attention-ffn-input`.
- Added `tools/ds4-v100-tp-ep-true-attn-http-ab.py`.
- Forced route-plan async upload off for this path because the routed FFN norm
  input gate is incompatible with the async route-plan uploader.

## Next

The next TP/EP work should target the new measured bottleneck, not PP:

1. Make post-attention serving memory-admitted at `32` slots / `256K`.
2. Replace the current attention-output projection/gather path with the intended
   TP collective/kernel structure.
3. Keep the post-attention path default-off until readiness passes and a quality
   baseline is accepted.
