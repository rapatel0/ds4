# Sprint 107 F8 Grouped DS4 Fast Runs

Cluster target: `llamacpp-build-8gpu` on `gpu-01`.

Build:

```text
cd /workspace/ds4
make tools/ds4-v100-replay tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_projection_attention_smoke \
  tests/cuda_v100_stage_scheduler_smoke tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke CUDA_ARCH=sm_70 -j80
```

Correctness:

```text
./tests/cuda_source_dtypes_smoke
./tests/cuda_v100_projection_attention_smoke
./tests/cuda_v100_stage_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --stage 0 --slots 4
./tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-s090 --slots 8
./tests/cuda_v100_selected_token_smoke --model /models/DSv4-Flash-256e-fixed.gguf --appliance-dir /workspace/ds4-appliance-full-tm-s090 --expected-token-hex 3136
```

Throughput:

| Run | Summary |
|---|---|
| `soak-8slot-fast` | `31.811137` generated tok/s, `29.822941` continuation tok/s, `8/8` token match |
| `soak-8slot-fast-repeat` | `31.630774` generated tok/s, `29.653851` continuation tok/s, `8/8` token match |
| `soak-8slot-generic` | `31.098630` generated tok/s, `29.154965` continuation tok/s, `8/8` token match |
| `soak-4slot-fast` | `20.095510` generated tok/s, `18.839541` continuation tok/s, `4/4` token match |
| `soak-4slot-generic` | `20.105807` generated tok/s, `18.849194` continuation tok/s, `4/4` token match |

Rollback:

```text
DS4_V100_CUDA_F8_GROUPED_DS4_FAST=0
```
