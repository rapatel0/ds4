# TEMP Status Report 081

Date: 2026-05-25

## Current Focus

TP/EP-only serving work. Sprint 369 added durable utilization sampling to the
permanent TP/EP profile harness so future serving and direct-token-major
experiments can carry GPU utilization evidence in the same artifact as tok/s,
coalescing, and compressed-KV timings.

## What Changed

- Added `--gpu-sample-interval-ms` to
  `tools/ds4-v100-tp-ep-profile.py`.
- Default remains `0`, so there is no sampler thread or `nvidia-smi` polling
  overhead unless explicitly enabled.
- When enabled, the harness writes:
  - `gpu_util.csv`
  - `gpu_sample_count`
  - `gpu_util_avg`
  - `gpu_util_max`
  - `gpu_mem_used_max_mib`
  - `gpu_per_gpu`
- The sampler works for both HTTP serving profiles and direct token-major
  profiles.

## Validation

Local:

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`: pass
- `bash -n tools/ds4-v100-run-appliance.sh`: pass
- `./ds4_test --server`: pass
- `./ds4_test --metal-kernels`: pass

V100 pod:

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`: pass
- `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`: pass
- Sampled TP/EP `/v1/chat/completions` smoke:
  - shape: `32` configured slots, `4` concurrent requests, `4` tokens/request,
    `256K` context, `position=100000`
  - HTTP 200: `4/4`
  - coalesced batch size: `4`
  - generated tokens: `16`
  - first token: `46915`
  - server generated decode tok/s: `99.340235`
  - client generated tok/s: `1.739606` because the small run includes model
    startup and HTTP orchestration
  - GPU samples: `264`
  - average GPU utilization: `8.412879%`
  - max GPU utilization: `39%`
  - per-GPU avg utilization:
    - GPU0: `27.090909%`
    - GPU1: `3.454545%`
    - GPU2: `3.424242%`
    - GPU3: `3.393939%`
    - GPU4: `3.393939%`
    - GPU5: `7.636364%`
    - GPU6: `7.212121%`
    - GPU7: `11.696970%`

Cluster artifact:

`/workspace/logs/sprint369-gpu-sampler-smoke/none-hc-stream-sync`

Local copy:

`logs/from-cluster/sprint369-gpu-sampler-smoke/none-hc-stream-sync`

## Current Interpretation

The harness now makes low utilization visible in the normal profile artifact.
The 4-active-request smoke confirms an imbalance pattern: GPU0 is materially
busier than the other ranks even though the run uses the TP/EP serving path.
The next optimization sprint should run a sampled active-slot matrix and then
target active-slot compaction, dense projection/state fragmentation, or EP
balance with the utilization evidence attached.
