# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `32` requests, `32` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `False`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-blocked**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `108.084959` | `20.315962` | `0.1879628968541312` |
| server continuation decode tok/s | `107.964491` | `20.308358` | `0.1881021974160004` |
| client generated tok/s | `17.810479266520147` | `8.69354826447401` | `0.4881142239005327` |
| avg GPU util % | `4.198033707865169` | `7.124305555555556` | `1.6970577301959036` |
| min free VRAM MiB | `2106` | `1328` | `0.630579297245964` |
| attention output ms | `0.0` | `512.62943` | `None` |
| post-attn FFN input ms | `0.0` | `144.063057` | `None` |
| attention projection ms | `54.322579` | `58.002707` | `1.0677458262797133` |
| attention state ms | `41.194539` | `44.987935` | `1.0920849241691963` |
| compressed KV ms | `72.464854` | `80.341195` | `1.1086918770304843` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[122445, 121932, 89368, 121932, 36155, 60846, 94562, 35717, 54491, 35119, 44742, 38909, 64780, 128819, 47710, 114963, 56697, 48061, 28264, 62673, 60041, 10257, 124155, 65417, 32974, 128816, 15984, 70623, 41456, 67132, 123477, 87337]`

## Artifacts

- Control: `/workspace/logs/sprint411-true-attn-http-ab-rerun/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint411-true-attn-http-ab-rerun/candidate/none-hc-nccl-allgather-hc-stream-sync-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
