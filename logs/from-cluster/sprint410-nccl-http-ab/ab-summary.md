# DS4 V100 TP/EP HC-Current NCCL HTTP A/B

- Shape: `32` requests, `32` slots, `262144` ctx, `32` generated tokens/request
- Control ready: `True`
- Candidate ready: `True`
- Parity match: `True` (`32/32` pairs)
- Decision: **promote-hc-current-nccl**

| Metric | Control | HC-current NCCL | Candidate/control |
|---|---:|---:|---:|
| server generated decode tok/s | `101.89789` | `107.723452` | `1.05717058518091` |
| server continuation decode tok/s | `101.682616` | `107.545644` | `1.0576600822307718` |
| client generated tok/s | `17.223946779516098` | `16.62711980696785` | `0.9653490004243375` |
| avg GPU util % | `4.535714285714286` | `3.5242718446601944` | `0.7770048161455547` |
| max GPU util % | `49.0` | `47.0` | `None` |
| min free VRAM MiB | `2738` | `2106` | `None` |
| post-close NCCL free MiB | `None` | `2240` | `None` |
| HC-current gather ms | `3.279789` | `5.700894` | `1.7381892554673486` |
| HC-current input ms | `265.428902` | `254.759223` | `0.9598021205693719` |

## Artifacts

- Control: `/workspace/logs/sprint410-nccl-http-ab/control/none-model-router-compact-moe`
- Candidate: `/workspace/logs/sprint410-nccl-http-ab/candidate/none-hc-nccl-allgather-hc-stream-sync-model-router-compact-moe`
- Parity: `/workspace/logs/sprint410-nccl-http-ab/response-parity.json`
