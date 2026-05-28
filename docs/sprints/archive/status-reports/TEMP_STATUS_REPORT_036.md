# TEMP_STATUS_REPORT_036

Date: 2026-05-24

## Topline

Sprint 324 shipped the first bounded TP/EP compressed-row storage and
raw+compressed attention-read path.

Implemented in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

- Resident per-rank attention compressed state/cache buffers.
- Resident per-rank indexer compressed state/cache buffers.
- Gathered TP compressor/indexer projection shards.
- Compressor state store with APE.
- Emit-boundary pooling, RMSNorm, RoPE, F16 round-trip, and ratio-4 shift.
- Raw+compressed attention read when a compressed row is visible.
- Bounded ratio-4 indexer score/top-k for the single visible row.

This remains bounded to one visible compressed row. It proves the lifecycle and
attention merge path, not full long-history compressed-cache parity.

## V100 Validation

Build:

- Command: `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

32-slot / 256K all-layer smoke:

- Position: `262143`
- Layers: `43`
- Result: PASS
- `pass_invocations=43`
- `sum_decode_ms=1670.069087`
- `projected_slot_step_tok_s=19.160884`
- Ratio-128 layers report compressed rows visible in the attention read.
- Ratio-4 layers report `indexer_topk_count=512` and compressed rows selected.

HTTP parity:

- Case: `short_reasoning_plain`
- Result: FAIL, but the request completed.
- Expected: `16`
- Actual: `mere`
- Generated token: `88445`
- Wall tok/s: `20.366798`
- Decode tok/s: `21.214211`

## Interpretation

The new compressed-row/read path is active and structurally stable at the
target `32` slot / `256K` shape. It does not solve reference parity yet.

The most likely next correctness step is not more endpoint work. It is a
layer-local comparison against the non-TP reference path:

- emitted attention compressed row
- emitted indexer compressed row
- indexer q/w score
- selected compressed row
- raw+compressed attention output

Layer `2` at a ratio-4 emit position is the best first target.

## Artifacts

- `logs/from-cluster/sprint324-compressed-row-storage-v2/cluster/all-layer-smoke.log`
- `logs/from-cluster/sprint324-compressed-row-storage-v2/cluster/http-parity/`
