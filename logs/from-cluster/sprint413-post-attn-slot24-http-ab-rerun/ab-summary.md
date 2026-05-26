# DS4 V100 TP/EP True-Attention HTTP A/B

- Shape: `24` requests, `24` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `True`
- Candidate active: `True`
- Decision: **true-attention-post-attention-serving-operational**

| Metric | Control | True-attn candidate | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `90.974366` | `19.716583` | `0.21672679752448068` |
| server continuation decode tok/s | `90.772903` | `19.843168` | `0.21860232893510081` |
| client generated tok/s | `14.944926659484418` | `7.367808881719989` | `0.4929973260888635` |
| avg GPU util % | `3.8363095238095237` | `6.382352941176471` | `1.6636699675991422` |
| min free VRAM MiB | `3198` | `2428` | `0.7592245153220762` |
| attention output ms | `0.0` | `365.433075` | `None` |
| post-attn FFN input ms | `0.0` | `115.793207` | `None` |
| attention projection ms | `46.411821` | `49.686244` | `1.0705514873032023` |
| attention state ms | `34.303325` | `35.898697` | `1.0465077947983177` |
| compressed KV ms | `69.220566` | `53.305449` | `0.7700810912178904` |

## Response 0

- Control sequence: `[83480, 79768, 46915, 98751, 97212, 2460, 97466, 69260, 31128, 13655, 90448, 32974, 94704, 78827, 117465, 109502, 123327, 92132, 57045, 13698, 87562, 73669, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484, 83484]`
- Candidate sequence: `[123477, 104015, 119333, 121932, 27525, 121932, 57648, 124436, 123327, 109865, 123477, 7447, 62433, 120993, 92132, 56697, 112864, 56697, 25903, 60432, 121932, 92132, 50319, 121932, 115959, 118235, 123477, 101267, 5801, 80944, 121932, 85842]`

## Artifacts

- Control: `/workspace/logs/sprint413-post-attn-slot24-http-ab-rerun/control/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint413-post-attn-slot24-http-ab-rerun/candidate/none-hc-nccl-allgather-hc-stream-sync-attention-output-nccl-allgather-post-attention-ffn-input-model-router-no-route-plan-async-upload-compact-moe`
