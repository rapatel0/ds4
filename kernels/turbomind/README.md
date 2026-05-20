# TurboMind MXFP4 Grouped GEMM Source Copy

This directory contains the TurboMind C ABI wrapper and lmdeploy `turbomind`
support source copied from the local DeepSeek/llama.cpp working tree:

```text
/Users/ravi/repos/deepseek/ggml/vendor/turbomind
/Users/ravi/repos/deepseek/research/lmdeploy/src/turbomind
```

Copied during Sprint 081 from source repo commit
`5903432d826b7b10cdc6d02d8d5da1bbe65371b8`.

The intended proof is:

```bash
cmake -S kernels/turbomind/ggml-turbomind \
  -B build/turbomind-v100 \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/turbomind-v100 --target ggml-turbomind test_ggml_turbomind_grouped_compare -j 8
./build/turbomind-v100/test_ggml_turbomind_grouped_compare \
  ./build/turbomind-v100/libggml-turbomind.so
```

This is copied source, not a build-time dependency on `~/repos/deepseek`.
The copied CMake default points `LMDEPLOY_SRC` at
`kernels/turbomind/lmdeploy/src`.
