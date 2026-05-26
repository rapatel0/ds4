# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `28` requests, `28` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `True`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-operational**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `97.486879` | `19.70859` | `0.20216659105478185` |
| server continuation decode tok/s | `97.444398` | `19.693809` | `0.20210303931478954` |
| client generated tok/s | `14.530917752011` | `7.543522997079557` | `0.5191360329622383` |
| avg GPU util % | `2.9325` | `5.844230769230769` | `1.9929175683651386` |
| min free VRAM MiB | `2566` | `1790` | `0.6975837879968823` |
| attention output ms | `0.0` | `460.797268` | `None` |
| post-attn FFN input ms | `0.0` | `129.119597` | `None` |
| attention projection ms | `55.879071` | `49.550538` | `0.8867459160156761` |
| attention state ms | `39.388492` | `36.933089` | `0.9376619191209453` |
| compressed KV ms | `79.900473` | `73.616155` | `0.9213481752479739` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[32461, 124727, 73288, 123477, 107880, 63104, 95158, 32974, 58572, 32974, 33611, 26343, 1853, 123477, 110614, 70623, 64811, 118187, 3090, 23824, 14868, 39913, 6256, 50615, 27623, 32461, 43048, 90042, 128818, 117160, 25689, 91569]`

## Artifacts

- Control: `/workspace/logs/sprint414-semantic-noskip-28slot-http-ab/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint414-semantic-noskip-28slot-http-ab/candidate/none-hc-nccl-allgather-hc-stream-sync-attention-output-nccl-allgather-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
