---
sprint: 365
title: TP/EP Fused Local Compressed Input Fill
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 365 - TP/EP Fused Local Compressed Input Fill

## Overview

Sprint 364 proved direct peer-read input fill is the wrong direction. The
correct constraint is to preserve local per-rank current data, then reduce
local launch/read duplication.

The current compressed projection path fills attention compressor KV and gate
half-input buffers with two separate kernels from the same local current
vector. Ratio-4 can optionally fuse all five compressor/indexer fills, but
ratio-128 layers still use separate attention fills, and the fused ratio-4
path is tied to indexer enablement.

This sprint adds a smaller local-only diagnostic gate:

```text
local current -> attn_compress_kv half input
local current -> attn_compress_gate half input
```

becomes:

```text
local current -> attn_compress_kv + attn_compress_gate half inputs
```

This is TP/EP-only. No PP/layer-split work. No MTP.

## Implementation

1. Add a fused two-destination local half-fill CUDA kernel for attention
   compressor KV/gate inputs.
2. Add `--true-ds4-compressed-kv-fused-attn-input-fill-gate`.
3. Expose it through:
   - `tools/ds4-v100-run-appliance.sh`,
   - `deploy/v100/ds4-v100-appliance.env.example`,
   - `tools/ds4-v100-tp-ep-profile.py`.
4. Keep the existing ratio-4 five-way fused input fill separate.
5. Keep this new gate default-off unless V100 evidence supports promotion.

## Verification

- Local syntax checks pass.
- V100 `sm_70` build passes.
- V100 emitted-row A/B at `32` slots / `256K`:
  - control: production pool+norm default,
  - candidate: fused local attention input fill plus production pool+norm.
- Both variants preserve finite output head and first selected token.
- Compare generated decode tok/s, compressed-KV sum, and
  `attn_input_fill_ms`.

## Definition of Done

- [x] The fused local fill kernel compiles for `sm_70`.
- [x] The gate is reachable from direct smoke, launcher, and profile harness.
- [x] V100 A/B passes correctness invariants.
- [x] Results are summarized in this sprint doc, `STATUS.md`, and `VISION.md`.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Implemented the local fused attention compressor input-fill diagnostic:

- `--true-ds4-compressed-kv-fused-attn-input-fill-gate`,
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL=1`,
- `tools/ds4-v100-tp-ep-profile.py --fused-compressed-attn-input-fill`.

The gate preserves the local staged per-rank current vector, then replaces the
two local attention compressor half-fill launches with one kernel that writes
both `attn_compress_kv` and `attn_compress_gate` inputs.

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

One-token emitted-row same-build A/B at `32` slots / `256K` /
`position=262143`:

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Attn input fill | Indexer input fill | Fused attn rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pool+norm default | 54639 | 0 | 76.560441 | 20.099045 | 132.925686 ms | 12.811472 ms | 3.357521 ms | 0 |
| fused attn input fill | 54639 | 0 | 80.160077 | 20.861285 | 126.795906 ms | 12.237819 ms | 4.102924 ms | 41 |

Full 32-step direct A/B at `32` slots / `256K` / `position=262112`:

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Attn input fill | Indexer input fill | Fused attn rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pool+norm default | 98751 | 0 | 94.237924 | 73.546958 | 3532.911129 ms | 403.084915 ms | 128.819697 ms | 0 |
| fused attn input fill | 98751 | 0 | 94.396298 | 73.623862 | 3499.213977 ms | 391.472748 ms | 130.015176 ms | 1312 |

Selected-token HTTP A/B at the same `32` slot / `256K` / `position=262112`
long-context window:

| Variant | HTTP 200 | First token | Finite bad | Client tok/s | Compressed-KV sum | Attn input fill | Indexer input fill | Fused attn rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| launcher default | 32/32 | 109328 | 0 | 72.886325 | 3493.666516 ms | 403.878494 ms | 128.585626 ms | 0 |
| fused attn input fill | 32/32 | 109328 | 0 | 70.674037 | 3506.331429 ms | 393.947826 ms | 131.366994 ms | 1312 |

Local validation:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
./ds4_test --server
./ds4_test --metal-kernels
```

All passed. Full local `make test` still requires a local `ds4flash.gguf` and
fails without that model file.

## Decision

Keep fused local attention input fill as a diagnostic-only gate. It is
correct, and the direct 32-step run slightly improves decode
(`94.237924` to `94.396298` tok/s) and compressed-KV sum
(`3532.911129` to `3499.213977` ms), but the selected-token HTTP run regresses
client throughput (`72.886325` to `70.674037` tok/s) and compressed-KV sum
(`3493.666516` to `3506.331429` ms).

The next TP/EP optimization should target the larger compressed/indexer dense
projection or attention projection/state boundary, not more input-fill
micro-fusion.

Artifacts:

```text
logs/from-cluster/sprint365-fused-attn-input-fill-smoke/
logs/from-cluster/sprint365-fused-attn-input-fill/
logs/from-cluster/sprint365-fused-attn-input-fill-http/
```
