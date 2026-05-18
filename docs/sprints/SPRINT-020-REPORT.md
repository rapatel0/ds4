# SPRINT-020 Report: V100 Compressor/Indexer Descriptors And HC Bridge

## Verdict

`EXTEND`.

Sprint 020 shipped the descriptor and HC halves of the planned bridge. The V100
runtime now binds real DS4 attention-compressor and ratio-4 indexer tensors in
`ds4_v100_layer_state`, and the integrated layer executor has a DS4 HC-state
entrypoint that accepts `[4 x 4096]`, runs HC attention pre/post, runs HC FFN
pre/post, and passes on the V100 cluster.

Executor-owned compressed-row generation remains open. The integrated layer
still accepts test-provided compressed KV rows, so Sprint 020 does not claim a
full `SHIP`.

## What Changed

- Added compressor and indexer descriptor ownership to `ds4_v100_layer_state`.
- Added BF16 matrix view support for source BF16 compressor/indexer projection
  tensors.
- Extended the layer-state smoke to validate ratio, compressor width, and
  ratio-4 indexer dimensions from the real pack index.
- Refactored `ds4_v100_layer_execute_decode` so attention output and FFN delta
  are separate reusable bodies.
- Added `ds4_v100_layer_execute_hc_decode` for exact DS4 HC pre/post placement:
  attention HC pre, attention body, attention HC post, FFN HC pre, FFN body,
  FFN HC post.
- Extended `tests/cuda_v100_integrated_layer_smoke.c` to execute the HC entrypoint
  on V100 and validate finite/nonzero HC output plus selected expert ranges.
- Updated the integrated smoke to use the file-descriptor-backed CUDA model
  cache for large source-layout F32 HC projection weights.

## Validation

Local checks:

```sh
make ds4_v100_layer_state.o ds4_v100_layer_execute.o tests/cuda_v100_integrated_layer_smoke.o tests/v100_layer_state_smoke
./tests/v100_layer_state_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --layer 2
git diff --check
```

Layer-state descriptor smoke:

```text
v100_layer_state_smoke: layer=2 stage=0 gpu=0 class=ratio_4 router=hash hidden=4096 q=32768 kv=512 ratio=4 comp=1024 index_q=8192 mid=2048 experts=256 ffn_span=11784434944 attn_span=8329207040 ok
```

One-card V100 integrated HC smoke:

```text
cuda_v100_integrated_layer_smoke: layer=2 token=16 pos=16 expert0=84 hc=ok gpu=0 arena_bytes=11784434944 hidden=4096 raw=3 comp=3 ok
```

Full 8-GPU V100 appliance gate:

```text
gate	integrated_layer	PASS	command=./tests/cuda_v100_integrated_layer_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --layer 2 --router-token 16 --position 16
gate	readiness	NOT_READY	missing=full_43_layer_scheduler,real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Logs:

- `docs/sprints/drafts/SPRINT-020-GATE-CLUSTER-8GPU.log`
- `docs/sprints/drafts/SPRINT-020-GATE-CLUSTER-8GPU/`

## Important Limits

- The executor still receives `compressed_kv` and optional masks from the test.
  It does not yet generate attention-compressor rows, indexer rows, or top-k
  visibility inside `ds4_v100_layer_execute`.
- HC output is checked for device execution, finite values, and valid routing.
  It is not yet compared against a full CPU HC reference vector.
- This is still a representative layer-2 ratio-4 proof, not a 43-layer decode.
- No MTP, serving endpoint, slot wavefront, or throughput benchmark shipped in
  this sprint.

## Next Sprint

Sprint 021 should make compressed-row state executor-owned: add decode config
for raw/comp/indexer caches and compressor state tensors, project BF16
compressor KV/score rows from `attn_norm`, update attention and indexer
compressor state, run ratio-4 indexer scoring/top-k when needed, and feed
indexed compressed attention without dense masks.
