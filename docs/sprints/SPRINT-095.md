# Sprint 095 - Request Rendezvous And F8 Cache Probe

Date: 2026-05-20

## Objective

Stabilize practical multi-slot serving by preventing avoidable split batches,
then test whether an opt-in F8-to-F16 arena cache can move dense/shared F8
matmul onto the existing V100 FP16 cuBLAS tensor-core path.

## Changes

- Added `--microbatch-wait-us` to `tools/ds4-v100-replay`.
  - CLI default remains `5000` us to preserve direct runtime behavior.
  - `DS4_V100_MICROBATCH_WAIT_US=auto` in the appliance launcher resolves to
    `50000` us when `DS4_V100_ACTIVE_MICROBATCH > 1`, otherwise `0`.
  - Status JSON and Prometheus metrics now expose the resolved coalescing wait.
- Added `DS4_V100_MICROBATCH_WAIT_US=auto` to the example env, k8s config, and
  appliance soak harness.
- Added an opt-in F8 E4M3 B128 arena cache behind `DS4_CUDA_F8_F16_CACHE=1`.
  - The cache dequantizes selected F8 arena matrices to resident FP16 buffers
    and uses `cublasGemmEx` for batched F8 matmul calls.
  - It is intentionally not a default because the measured path is flat.

## Validation

All cluster runs used pod `llamacpp-build-8gpu`, repo
`/workspace/ds4-sprint082`, appliance
`/workspace/ds4-appliance-full-tm-s090`, and the copied TurboMind library.

| Case | Result |
| --- | --- |
| Build | `make -C /workspace/ds4-sprint082 tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8` passed |
| 4-slot, 1M, rendezvous auto | `token_match=4/4`, `12.597711` generated tok/s, `11.810354` continuation tok/s |
| 8-slot, 256K, rendezvous auto | `token_match=8/8`, `17.052974` generated tok/s, `15.987163` continuation tok/s |
| 4-slot, 1M, F8-F16 cache on | `token_match=4/4`, `12.614479` generated tok/s, `11.826074` continuation tok/s |
| Decode-window `nvprof` | F8 arena matmul `42.52%`, HtoD `31.48%`, TurboMind GEMM `13.82%` |

Artifacts:

- `logs/from-cluster/sprint095-rendezvous/soak-4slot-auto`
- `logs/from-cluster/sprint095-rendezvous/soak-8slot-256k-auto`
- `logs/from-cluster/sprint095-rendezvous/soak-4slot-cache-auto`
- `logs/from-cluster/sprint095-rendezvous/profile`

## Decision

Ship the microbatch rendezvous control as the production default through the
launcher. The 8-slot/256K profile now passes cleanly; this fixes the split-batch
behavior seen in earlier probes.

Keep the F8-F16 arena cache as an experimental opt-in only. It is correct, but
does not materially improve the measured 4-slot/1M appliance path.

## Next

- Attack the real profiler buckets: F8 projection/shared matmul launch shape and
  residual HtoD control copies.
- Add a targeted Nsight Compute or `nvprof` harness for a representative batched
  F8 projection launch, not just a full decode summary.
- Improve continuous multi-token serving state so higher active slot counts
  raise effective GPU work instead of repeatedly resetting per request.
