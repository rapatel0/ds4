# Sprint 101 Cluster Runs

Date: 2026-05-20

Cluster: `llamacpp-build-8gpu` on `gpu-01`

Workspace: `/workspace/ds4-sprint082`

Appliance: `/workspace/ds4-appliance-full-tm-s090`

TurboMind library:
`/workspace/ds4-sprint082/build/turbomind-v100/libggml-turbomind.so`

Note: `runtime/startup.env` records launcher-resolved defaults, but not the
outer shell variable used to enable the Sprint 101 opt-in. Directory names are
therefore part of the run contract:

- `soak-*-default`: `DS4_V100_ENABLE_BATCH_ATTN_PROJ` unset.
- `soak-*-batch`: `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1`.

## Results

| Run | Context | Slots | Batch Projection | Generated tok/s | Continuation tok/s | Token Match |
|---|---:|---:|---|---:|---:|---:|
| `soak-4slot-default` | 1,048,576 | 4 | off | 18.102742 | 16.971321 | 4/4 |
| `soak-4slot-batch` | 1,048,576 | 4 | on | 17.503345 | 16.409386 | 4/4 |
| `soak-8slot-default` | 262,144 | 8 | off | 26.402101 | 24.751970 | 8/8 |
| `soak-8slot-batch` | 262,144 | 8 | on | 26.432087 | 24.780082 | 8/8 |

Decision: keep `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` opt-in.
