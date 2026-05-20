# Sprint 102 Cluster Runs

Date: 2026-05-20

Cluster: `llamacpp-build-8gpu` on `gpu-01`

Workspace: `/workspace/ds4-sprint082`

Appliance: `/workspace/ds4-appliance-full-tm-s090`

TurboMind library:
`/workspace/ds4-sprint082/build/turbomind-v100/libggml-turbomind.so`

## Results

| Run | Context | Slots | F8 Row-Pair | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---|---:|---:|---:|
| `soak-4slot-default` | 1,048,576 | 4 | off | 17.821073 | 16.707256 | 4/4 |
| `soak-4slot-rowpair` | 1,048,576 | 4 | on | 18.500281 | 17.344013 | 4/4 |
| `soak-8slot-default` | 262,144 | 8 | off | 26.447308 | 24.794352 | 8/8 |
| `soak-8slot-rowpair` | 262,144 | 8 | on | 27.037514 | 25.347670 | 8/8 |
| `soak-8slot-launcher-default` | 262,144 | 8 | on | 27.049799 | 25.359186 | 8/8 |

`soak-8slot-launcher-default/runtime/startup.env` records
`DS4_V100_CUDA_F8_ROWPAIR=1`, proving the row-pair path is active through the
operator launcher default rather than only an outer shell variable.

Decision: ship `DS4_V100_CUDA_F8_ROWPAIR=1` as the production appliance default.
