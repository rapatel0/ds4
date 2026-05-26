# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `28` requests, `28` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `True`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-operational**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `98.358976` | `31.091919` | `0.316106574757346` |
| server continuation decode tok/s | `98.279705` | `31.06439` | `0.31608143308936465` |
| client generated tok/s | `15.621161353928729` | `10.366506446092782` | `0.663619446161447` |
| avg GPU util % | `3.8579545454545454` | `7.899436090225564` | `2.0475710631512505` |
| min free VRAM MiB | `2566` | `1790` | `0.6975837879968823` |
| attention output ms | `0.0` | `19.520681` | `None` |
| post-attn FFN input ms | `0.0` | `82.034351` | `None` |
| attention projection ms | `51.437702` | `49.908197` | `0.9702649041358807` |
| attention state ms | `36.617314` | `37.77405` | `1.0315898648382567` |
| compressed KV ms | `72.73397` | `74.368809` | `1.0224769664023565` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[32461, 124727, 73288, 123477, 107880, 63104, 95158, 32974, 58572, 32974, 33611, 26343, 1853, 123477, 110614, 70623, 64811, 118187, 3090, 23824, 14868, 39913, 6256, 50615, 27623, 32461, 43048, 90042, 128818, 117160, 25689, 91569]`

## Artifacts

- Control: `/workspace/logs/sprint414-semantic-skip-stats-28slot-http-ab/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint414-semantic-skip-stats-28slot-http-ab/candidate/none-hc-nccl-allgather-hc-stream-sync-attention-output-nccl-allgather-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
