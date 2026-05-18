# SPRINT-022 Report: Bias Router And Resident Stage Scheduler

## Verdict

`SHIP`.

Sprint 022 removed the hash-router-only execution limit and added the first
resident multi-layer scheduler surface. The V100 path now validates both router
families and executes gpu0 layers 0-5 from a token embedding seed with full
stage-resident pack bytes.

## What Changed

- Generalized `execute_ffn_delta` so it chooses hash or bias router metadata
  from `ds4_v100_layer_state`.
- Extended the integrated V100 smoke with a CPU bias-router reference for
  selected experts and route weights.
- Added `ds4_v100_stage_scheduler`, a reusable resident stage scheduler for the
  token-embedding stage.
- The scheduler opens the real context and pack index, uploads all gpu0 pack
  entries into one resident arena, initializes local layer states, allocates
  per-layer decode caches, seeds `[4 x 4096]` HC from `token_embd.weight`, and
  walks layers 0-5 with `ds4_v100_layer_execute_hc_decode`.
- Added `tests/cuda_v100_stage_scheduler_smoke`.
- Added `integrated_layer_bias` and `stage_scheduler` to
  `tools/ds4-v100-gate.sh`.

## Validation

Local checks:

```sh
make ds4_v100_layer_execute.o tests/cuda_v100_integrated_layer_smoke.o
make ds4_v100_scheduler.o tests/cuda_v100_stage_scheduler_smoke.o
git diff --check
```

One-card V100 layer 2 regression:

```text
cuda_v100_integrated_layer_smoke: layer=2 token=16 pos=16 expert0=84 hc=ok cache=ok gpu=0 arena_bytes=11784434944 hidden=4096 raw=3 comp=3 ok
```

One-card V100 layer 3 bias-router gate:

```text
cuda_v100_integrated_layer_smoke: layer=3 token=16 pos=16 expert0=45 hc=ok cache=skip gpu=0 arena_bytes=15356173824 hidden=4096 raw=3 comp=3 ok
```

Resident stage scheduler:

```text
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 token=16 pos=16 arena_bytes=22524134668 uploaded_tensors=173 uploaded_bytes=22524130064 expert0=108 ok
```

Full V100 appliance gate:

```text
gate	integrated_layer_bias	PASS	command=./tests/cuda_v100_integrated_layer_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --layer 3 --router-token 16 --position 16
gate	stage_scheduler	PASS	command=./tests/cuda_v100_stage_scheduler_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --stage 0 --token 16 --position 16
gate	readiness	NOT_READY	missing=full_43_layer_scheduler,real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Logs:

- `docs/sprints/drafts/SPRINT-022-GATE-CLUSTER-8GPU/summary.log`
- `docs/sprints/drafts/SPRINT-022-GATE-CLUSTER-8GPU/integrated_layer.log`
- `docs/sprints/drafts/SPRINT-022-GATE-CLUSTER-8GPU/integrated_layer_bias.log`
- `docs/sprints/drafts/SPRINT-022-GATE-CLUSTER-8GPU/stage_scheduler.log`

## Important Limits

- The scheduler is stage-local and currently supports the token-embedding
  stage. Cross-GPU HC relay into stages 1-7 is not wired into the production
  scheduler yet.
- The final output-head selected-token gate is not connected.
- The stage scheduler uses executor-native F32 cache tensors, matching the
  current layer executor contract. Bridging to the context-owned F16 KV arena
  remains future work.
- Stage scheduler validation checks finite nonzero HC, layer count, and route
  reporting. It does not compare full hidden vectors to a CPU oracle.
- The stage walk includes one decode position. Longer cache progression remains
  covered only by the layer-2 integrated cache loop.

## Next Sprint

Sprint 023 should extend the scheduler from stage-local execution to the
full 8-GPU chain: stage 0 output handoff, peer-copy HC relay into stages 1-7,
resident arenas for all stages, final gpu7 output-head logits, and the first
real selected-token comparison.
