# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `30` requests, `30` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `True`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-operational**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `105.035348` | `21.08917` | `0.20078164543235483` |
| server continuation decode tok/s | `104.882275` | `21.069065` | `0.20088298999997853` |
| client generated tok/s | `16.197411761359966` | `8.21212433262282` | `0.5070022577442529` |
| avg GPU util % | `3.58375` | `6.425518134715026` | `1.792959367900949` |
| min free VRAM MiB | `2332` | `1556` | `0.6672384219554031` |
| attention output ms | `0.0` | `437.980246` | `None` |
| post-attn FFN input ms | `0.0` | `130.11834` | `None` |
| attention projection ms | `53.828568` | `51.609104` | `0.9587679166943472` |
| attention state ms | `38.409917` | `39.803595` | `1.0362843273001605` |
| compressed KV ms | `73.466251` | `74.552091` | `1.0147801199220035` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[115959, 65804, 94385, 109865, 76564, 102220, 39448, 108527, 126440, 47825, 94385, 98751, 27535, 50011, 52762, 57648, 121932, 123477, 65804, 57648, 43430, 113976, 58818, 86052, 57648, 57648, 108877, 98751, 123477, 27551, 118235, 113976]`

## Artifacts

- Control: `/workspace/logs/sprint413-post-attn-slot30-http-ab/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint413-post-attn-slot30-http-ab/candidate/none-hc-nccl-allgather-hc-stream-sync-attention-output-nccl-allgather-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
