# TEMP Status Report 077 - Sprint 365

Date: 2026-05-25

## Current Focus

TP/EP-only serving optimization. Sprint 365 tested whether a local
two-destination half-input fill for the attention compressor could reduce
compressed-KV launch/read fragmentation without repeating Sprint 364's slow
remote peer-read mistake.

## Implemented

- Added `--true-ds4-compressed-kv-fused-attn-input-fill-gate`.
- Added launcher env:
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL=0`.
- Added profile harness flag:
  `--fused-compressed-attn-input-fill`.
- Kept the gate default-off.

## Validation

Local checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
./ds4_test --server
./ds4_test --metal-kernels
```

Full local `make test` failed because the local `ds4flash.gguf` model file is
not present.

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Results

One-token emitted-row smoke, `32` slots / `256K` / `position=262143`:

| Variant | First token | Bad | Decode tok/s | Wall tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|
| pool+norm default | 54639 | 0 | 76.560441 | 20.099045 | 132.925686 ms |
| fused attn input fill | 54639 | 0 | 80.160077 | 20.861285 | 126.795906 ms |

Full direct 32-step A/B, `32` slots / `256K` / `position=262112`:

| Variant | First token | Bad | Decode tok/s | Wall tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|
| pool+norm default | 98751 | 0 | 94.237924 | 73.546958 | 3532.911129 ms |
| fused attn input fill | 98751 | 0 | 94.396298 | 73.623862 | 3499.213977 ms |

Selected-token HTTP A/B, same `32` slot / `256K` long-context window:

| Variant | HTTP 200 | First token | Bad | Client tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|
| launcher default | 32/32 | 109328 | 0 | 72.886325 | 3493.666516 ms |
| fused attn input fill | 32/32 | 109328 | 0 | 70.674037 | 3506.331429 ms |

## Decision

Do not promote this gate. It is correct and the direct run is slightly
positive, but the serving-visible HTTP gate regresses. Leave it as a
diagnostic option.

Next work should move up to a larger compressed/indexer dense projection or
attention projection/state boundary instead of continuing input-fill
micro-fusions.

## Artifacts

```text
logs/from-cluster/sprint365-fused-attn-input-fill-smoke/
logs/from-cluster/sprint365-fused-attn-input-fill/
logs/from-cluster/sprint365-fused-attn-input-fill-http/
```
