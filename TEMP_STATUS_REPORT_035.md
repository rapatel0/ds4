# TEMP Status Report 035 - Sprint 323 TP/EP Compressed-KV Projection Gate

Date: 2026-05-24

## Topline

Sprint 323 made concrete TP/EP progress but did not finish DS4 attention parity.

Implemented:

- `--true-ds4-compressed-kv-gate`
- `--true-ds4-indexer-attention-gate`
- BF16 resident dense binding through the existing FP16-cache/cuBLAS path
- Compressor/indexer projection execution for ratio-4 and ratio-128 layers
- Resident memory reductions needed for serving with output head:
  - freed unused dense float staging inputs in FP16-cache/cuBLAS mode
  - moved token embedding table off GPU0 and into host-backed row upload

## V100 Validation

Build:

- `make -j80 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

All-layer smoke:

- Shape: 32 slots / 256K context / 43 layers
- Gates:
  - `--true-ds4-post-attention-ffn-input-gate`
  - `--true-ds4-indexer-attention-gate`
- Result: PASS
- `tp_ep_compressed_kv_projection` rows: 43
- Ratio schedule observed:
  - SWA-only: 2 layers
  - ratio-4: 21 layers
  - ratio-128: 20 layers
- Final scaffold:
  - `pass_invocations=43`
  - `sum_decode_ms=1630.105625`
  - `projected_slot_step_tok_s=19.630630`

HTTP parity:

- Case: `short_reasoning_plain`
- Result: FAIL, but end-to-end request completed without OOM
- Expected: `16`
- Actual: `MARK`
- Generated token: `110609`
- Wall tok/s: `20.99589`
- Decode tok/s: `21.8601`

## Important Finding

The first HTTP parity attempt with compressor/indexer projection residency OOMed
near output-head/token-embedding startup. The root was not the compressor
projection outputs themselves; it was resident baggage:

- unused float input staging buffers retained for every resident dense op
- full BF16 token embedding table allocated on GPU0

After freeing unused dense staging buffers and replacing the full GPU0 token
embedding table with host-backed per-slot row uploads, the expanded TP path
served the parity request.

## Remaining Correctness Gap

The new gate executes compressor/indexer projections, but it does not yet
implement the full DeepSeek V4 compressed attention sequence:

- store emitted attention compressed rows
- store emitted indexer compressed rows
- compute indexer q/w scores against stored rows
- select ratio-4 top-k compressed rows
- run raw plus compressed attention softmax/value merge

That is the next TP-only task.

## Artifacts

- `logs/from-cluster/sprint323-compressed-kv-indexer/cluster/all-layer-smoke-input-free-v2.log`
- `logs/from-cluster/sprint323-compressed-kv-indexer/cluster/http-parity-v3/`
- `logs/from-cluster/sprint323-compressed-kv-indexer/cluster/http-parity-v4/`
