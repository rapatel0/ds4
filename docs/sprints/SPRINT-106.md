# Sprint 106 - Served Decode Baseline Profile

Date: 2026-05-20

## Objective

Capture a fresh warmed HTTP-serving profile of the Sprint 104 production
appliance baseline before making the next performance change.

## Context

Sprint 104 shipped F8 warp-reduction kernels and became the current committed
baseline:

- `31.451185` generated tok/s at 256K context, 8 slots.
- `20.026385` generated tok/s at 1M context, 4 slots.

Sprint 105 tested BF16/F32 warp reductions, but that change was rejected because
its repeat result stayed inside the Sprint 104 band. Sprint 106 therefore
restarted from measurement rather than applying another small reduction-order
change.

## Profile Run

Cluster target:

- pod: `llamacpp-build-8gpu`
- node: `gpu-01`
- storage: k8s-local `/workspace`
- host resources: 8x V100-SXM2-32GB, 80 CPU cores, 256 GB RAM

Served command under `nvprof`:

```text
./tools/ds4-v100-replay --serve \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --ctx 262144 --slots 8 --active-microbatch 8 \
  --microbatch-wait-us 50000 --queue-policy sequential \
  --tokens 16 --host 127.0.0.1 --port 18353 \
  --appliance-dir /workspace/ds4-appliance
```

The profiler window wrapped the warmed `/v100/selected-token` requests. All
eight responses returned HTTP 200 and the first token text hex `3136`.

## GPU Hot Path

| Bucket | GPU Time | Calls | Avg | Share |
|---|---:|---:|---:|---:|
| F8 rows2 arena matmul | 721.60 ms | 11,088 | 65.079 us | 38.97% |
| TurboMind SM70 MXFP4 GEMM | 470.85 ms | 4,752 | 99.083 us | 25.42% |
| F8 grouped rows2 arena matmul | 229.41 ms | 1,584 | 144.83 us | 12.39% |
| F32 matmul | 84.944 ms | 3,168 | 26.813 us | 4.59% |
| RMS norm plain | 55.123 ms | 3,168 | 17.399 us | 2.98% |
| Mixed attention decode | 53.808 ms | 1,584 | 33.969 us | 2.91% |
| BF16 matmul | 47.251 ms | 3,168 | 14.915 us | 2.55% |

F8 rows2 plus F8 grouped rows2 are still about `51.36%` of GPU time after the
Sprint 103 exact-bit decode and Sprint 104 warp-reduction changes. TurboMind is
now the second large bucket at `25.42%`.

## CUDA API Shape

| API | Time | Calls | Avg | Share |
|---|---:|---:|---:|---:|
| `cudaMemcpy` | 1.24935 s | 4,752 | 262.91 us | 68.14% |
| `cudaLaunchKernel` | 400.91 ms | 68,280 | 5.871 us | 21.87% |
| `cudaSetDevice` | 59.012 ms | 68,259 | 864 ns | 3.22% |
| `cudaDeviceSynchronize` | 34.575 ms | 297 | 116.41 us | 1.89% |

GPU memcpy activities were small: DtoH was `8.2045 ms` and HtoD was
`1.8367 ms`. The practical interpretation is that API overhead and launch count
are visible, but device traffic is not the dominant limiter in this warmed
decode profile.

## Decision

Do not spend the next sprint on more BF16/F32 reduction cleanups or VRAM-heavy
F8-to-F16 caches.

The next production change should target one of the two large remaining buckets:

1. F8 arena execution shape, especially reducing rows2/grouped rows2 work or
   launch count without changing source dtype semantics.
2. TurboMind routed expert execution, especially reducing per-layer grouped
   GEMM overhead or improving route batching/occupancy.

## Artifacts

- `logs/from-cluster/sprint106-profile-baseline-selected-token/nvprof-server.log`
- `logs/from-cluster/sprint106-profile-baseline-selected-token/server.log`
- `logs/from-cluster/sprint106-profile-baseline-selected-token/response_*.http`
