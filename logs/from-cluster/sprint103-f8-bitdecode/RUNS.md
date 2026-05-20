# Sprint 103 F8 Bit Decode Runs

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
| `soak-8slot` | `30.862791` generated tok/s, `28.933867` continuation tok/s, `8/8` token match |
| `soak-4slot` | `19.733742` generated tok/s, `18.500384` continuation tok/s, `4/4` token match |
