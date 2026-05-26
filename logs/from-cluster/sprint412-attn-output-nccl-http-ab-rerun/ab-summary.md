# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `32` requests, `32` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `False`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-served-reserve-blocked**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `101.539977` | `20.984393` | `0.20666139209387455` |
| server continuation decode tok/s | `101.187734` | `20.949901` | `0.2070399264005655` |
| client generated tok/s | `15.744130209984968` | `8.338640536946373` | `0.5296348814276184` |
| avg GPU util % | `2.8587962962962963` | `6.316287878787879` | `2.209422156790578` |
| min free VRAM MiB | `2106` | `1328` | `0.630579297245964` |
| attention output ms | `0.0` | `486.473759` | `None` |
| post-attn FFN input ms | `0.0` | `138.337609` | `None` |
| attention projection ms | `54.397643` | `54.613979` | `1.0039769370154512` |
| attention state ms | `42.933655` | `42.536043` | `0.9907389203178718` |
| compressed KV ms | `71.699447` | `54.112785` | `0.7547169087650006` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[122445, 121932, 89368, 121932, 36155, 60846, 94562, 35717, 54491, 35119, 44742, 38909, 64780, 128819, 47710, 114963, 56697, 48061, 28264, 62673, 60041, 10257, 124155, 65417, 32974, 128816, 15984, 70623, 41456, 67132, 123477, 87337]`

## Artifacts

- Control: `/workspace/logs/sprint412-attn-output-nccl-http-ab-rerun/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint412-attn-output-nccl-http-ab-rerun/candidate/none-hc-nccl-allgather-hc-stream-sync-attention-output-nccl-allgather-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
