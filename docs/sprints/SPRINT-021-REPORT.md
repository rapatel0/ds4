# SPRINT-021 Report: Executor-Owned Compressor/Indexer Decode Rows

## Verdict

`SHIP`.

Sprint 021 moved compressed-row generation into the V100 layer executor. Layer 2
now runs with mutable decoder-owned raw KV, attention-compressor state,
attention compressed rows, ratio-4 indexer state, ratio-4 indexer compressed
rows, indexer top-k scratch, and indexed mixed raw/compressed attention.

The path is still a representative layer-2 proof, not full selected-token
decode. The V100 appliance gate passes and correctly remains `ready=false`.

## What Changed

- Added `ds4_v100_layer_decode_cache` to carry raw KV, attention compressed KV,
  indexer compressed KV, compressor recurrence state, and top-k scratch.
- Added `decode_cache` to `ds4_v100_layer_execute_config` while preserving the
  explicit `raw_kv`/`compressed_kv` fixture path.
- Added executor validation for raw cache capacity, compressor state sizes,
  compressed-row capacity, ratio-4 indexer buffers, and top-k scratch.
- Projected attention compressor KV/score rows from real BF16
  `attn_compressor_kv` and `attn_compressor_gate` descriptors.
- Updated attention compressor recurrence through
  `ds4_gpu_compressor_update_tensor`, quantized emitted attention rows through
  the DS4 FP8 KV round-trip, and incremented row counts only at ratio
  boundaries.
- Projected ratio-4 indexer compressor rows from real BF16 descriptors and
  updated the indexer recurrence state.
- Projected `indexer_q` and indexer weights from real descriptors, ran
  indexer score/top-k, and selected
  `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` when indexed visibility
  is active.
- Extended the integrated V100 smoke to upload compressor/indexer matrices,
  execute eight decode-cache steps, force a small top-k threshold for coverage,
  and validate emitted attention/indexer rows plus top-k range.

## Validation

Local checks:

```sh
make ds4_v100_layer_execute.o tests/cuda_v100_integrated_layer_smoke.o
git diff --check
```

One-card V100 integrated smoke:

```text
cuda_v100_integrated_layer_smoke: layer=2 token=16 pos=16 expert0=84 hc=ok cache=ok gpu=0 arena_bytes=11784434944 hidden=4096 raw=3 comp=3 ok
```

Full V100 appliance gate:

```text
gate	integrated_layer	PASS	command=./tests/cuda_v100_integrated_layer_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --layer 2 --router-token 16 --position 16
gate	readiness	NOT_READY	missing=full_43_layer_scheduler,real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Logs:

- `docs/sprints/drafts/SPRINT-021-GATE-CLUSTER-8GPU/summary.log`
- `docs/sprints/drafts/SPRINT-021-GATE-CLUSTER-8GPU/integrated_layer.log`
- `docs/sprints/drafts/SPRINT-021-GATE-CLUSTER-8GPU/integrated-layer-smoke-gpu0-rerun.log`

## Important Limits

- This proves layer-2 ratio-4 executor-owned cache behavior. It does not walk
  all 43 layers.
- The integrated smoke forces `indexer_top_k=1` so indexed attention is reached
  after two emitted indexer rows. Production default remains 512 and still
  needs a longer stress test.
- Compressor/indexer scratch tensors are still allocated inside the executor
  call. Production scheduling should move these into reusable per-GPU scratch.
- HC output remains finite/range validated, not CPU-vector-referenced.
- No public server, MTP, multi-slot wavefront, or throughput benchmark shipped.

## Next Sprint

Sprint 022 should build the full 43-layer single-slot scheduler around the
executor-owned cache path: layer-class walking, per-layer decode-cache
allocation, HC handoff between layer owners, final output-head/top-k, and a
real selected-token gate.
