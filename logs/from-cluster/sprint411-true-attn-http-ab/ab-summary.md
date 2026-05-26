# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `32` requests, `32` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `False`
- Candidate active: `False`
- Decision: **true-attention-post-attention-serving-blocked**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `104.273108` | `0.0` | `0.0` |
| server continuation decode tok/s | `104.070843` | `0.0` | `0.0` |
| client generated tok/s | `17.99321489342456` | `0.0` | `0.0` |
| avg GPU util % | `4.2298850574712645` | `0.0` | `0.0` |
| min free VRAM MiB | `2106` | `1934` | `0.9183285849952516` |
| attention output ms | `0.0` | `None` | `None` |
| post-attn FFN input ms | `0.0` | `None` | `None` |
| attention projection ms | `56.995738` | `None` | `None` |
| attention state ms | `46.650732` | `None` | `None` |
| compressed KV ms | `72.271659` | `None` | `None` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `None`

## Artifacts

- Control: `/workspace/logs/sprint411-true-attn-http-ab/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint411-true-attn-http-ab/candidate/none-hc-nccl-allgather-hc-stream-sync-post-attention-ffn-input-model-router-compact-moe`
