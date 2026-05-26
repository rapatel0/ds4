# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `28` requests, `28` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `True`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-operational**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `100.245595` | `20.624419` | `0.20573890553495144` |
| server continuation decode tok/s | `100.036098` | `20.597899` | `0.20590466253491818` |
| client generated tok/s | `15.250440667793258` | `7.922564755228921` | `0.5194974314388331` |
| avg GPU util % | `3.47265625` | `6.273611111111111` | `1.8065741782277216` |
| min free VRAM MiB | `2566` | `1790` | `0.6975837879968823` |
| attention output ms | `0.0` | `422.604024` | `None` |
| post-attn FFN input ms | `0.0` | `128.822428` | `None` |
| attention projection ms | `53.208811` | `56.80871` | `1.0676560692175587` |
| attention state ms | `44.72416` | `40.858897` | `0.9135755037098517` |
| compressed KV ms | `69.952987` | `60.495492` | `0.8648021277490268` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[32461, 124727, 73288, 123477, 107880, 63104, 95158, 32974, 58572, 32974, 33611, 26343, 1853, 123477, 110614, 70623, 64811, 118187, 3090, 23824, 14868, 39913, 6256, 50615, 27623, 32461, 43048, 90042, 128818, 117160, 25689, 91569]`

## Artifacts

- Control: `/workspace/logs/sprint413-post-attn-slot28-http-ab/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint413-post-attn-slot28-http-ab/candidate/none-hc-nccl-allgather-hc-stream-sync-attention-output-nccl-allgather-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
