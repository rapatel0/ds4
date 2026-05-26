# TEMP Status Report 412

Current focus: TP/EP semantic serving path only. PP/layer-split remains frozen.

## Result

Sprint 412 tested attention-output NCCL inside the full post-attention serving
path.

Artifact:

```text
logs/from-cluster/sprint412-attn-output-nccl-http-ab-rerun/
```

Shape:

```text
32 concurrent HTTP requests
32 configured slots
262144 context
position 262080
32 generated tokens/request
HC-current NCCL enabled
attention-output NCCL enabled on candidate
lazy output head enabled
compact MoE decode enabled
model-router routes enabled
```

## Topline

| Metric | Control | Attention-output NCCL candidate |
|---|---:|---:|
| HTTP 200 | 32/32 | 32/32 |
| readiness | true | false |
| server generated decode tok/s | 101.539977 | 20.984393 |
| server continuation decode tok/s | 101.187734 | 20.949901 |
| client generated tok/s | 15.744130 | 8.338641 |
| avg sampled GPU util | 2.858796% | 6.316288% |
| min free VRAM | 2106 MiB | 1328 MiB |
| VRAM failures | 0 | 62 |
| attention output timer | 0.0 ms | 486.473759 ms |
| post-attn FFN input timer | 0.0 ms | 138.337609 ms |

Decision:
`true-attention-post-attention-serving-served-reserve-blocked`.

## Delta Versus Sprint 411

Attention-output NCCL helped slightly but did not solve the problem:

| Metric | Sprint 411 post-attn | Sprint 412 attn-output NCCL |
|---|---:|---:|
| server generated decode tok/s | 20.315962 | 20.984393 |
| attention output timer | 512.629430 ms | 486.473759 ms |
| post-attn FFN input timer | 144.063057 ms | 138.337609 ms |
| min free VRAM | 1328 MiB | 1328 MiB |
| VRAM failures | 62 | 62 |

## Code Change

Fixed `tools/ds4-v100-tp-ep-profile.py` so a truncated final
`tp_ep_token_major_scaffold` line cannot overwrite valid previously parsed
scaffold metrics with `null`.

## Next

Keep attention-output NCCL diagnostic-only. The next implementation should
target:

1. attention-output/post-attention scratch and temporary residency so the path
   passes the `1536 MiB` reserve at `32` slots / `256K`;
2. a purpose-built TP attention-output projection/gather kernel or collective
   shape rather than another narrow flag.
