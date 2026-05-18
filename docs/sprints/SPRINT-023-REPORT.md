# SPRINT-023 Report: Cross-GPU Two-Stage Scheduler Handoff

## Verdict

`SHIP`.

Sprint 023 added the first real cross-GPU scheduler hop. The runtime now opens
resident arenas for gpu0 and gpu1, executes layers 0-5 on gpu0, peer-copies HC
to gpu1, and executes layers 6-11 on gpu1.

## What Changed

- Added `ds4_gpu_set_device` to the GPU abstraction.
- CUDA tensors now remember their allocation device, and tensor
  fill/read/write/free/copy select the correct CUDA device.
- Cross-device tensor copy now uses `cudaMemcpyPeer`.
- CUDA model-range and model-arena caches are now device-aware. This fixed a
  real multi-GPU bug where a gpu0 cached control tensor could be reused by a
  gpu1 kernel.
- `ds4_v100_stage_scheduler` can open non-token stages.
- Added scheduler handoff and decode-from-HC APIs.
- Added `tests/cuda_v100_two_stage_scheduler_smoke`.
- Added `two_stage_scheduler` to the full V100 gate.

## Validation

Local checks:

```sh
make ds4_v100_scheduler.o tests/cuda_v100_two_stage_scheduler_smoke.o
git diff --check
```

Two-stage V100 smoke with launch blocking during diagnosis:

```text
cuda_v100_two_stage_scheduler_smoke: stage0=0-5 gpu0=0 stage1=6-11 gpu1=1 token=16 pos=16 uploaded0=22524130064 uploaded1=21494389008 expert0=108 expert1=161 ok
```

Full V100 appliance gate:

```text
gate	stage_scheduler	PASS	command=./tests/cuda_v100_stage_scheduler_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --stage 0 --token 16 --position 16
gate	two_stage_scheduler	PASS	command=./tests/cuda_v100_two_stage_scheduler_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --token 16 --position 16
gate	readiness	NOT_READY	missing=full_43_layer_scheduler,real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Logs:

- `docs/sprints/drafts/SPRINT-023-GATE-CLUSTER-8GPU/summary.log`
- `docs/sprints/drafts/SPRINT-023-GATE-CLUSTER-8GPU/stage_scheduler.log`
- `docs/sprints/drafts/SPRINT-023-GATE-CLUSTER-8GPU/two_stage_scheduler.log`

## Important Limits

- This proves the first 12 layers across gpu0 and gpu1. It does not yet walk
  all 43 layers.
- The scheduler still uses executor-native F32 KV/cache tensors rather than the
  context-owned F16 KV arena.
- The output head and selected-token oracle are still disconnected.
- Peer copy is synchronous and correctness-oriented. It has not been optimized
  with streams, double buffering, or FP16 relay compression.
- The CUDA model cache fix is intentionally conservative: cache entries are
  device-local. It may duplicate small control tensors across GPUs, which is
  acceptable for correctness.

## Next Sprint

Sprint 024 should generalize the two-stage chain into all eight stages and run
the full 43-layer scheduler without output-head selected-token yet. Once the
full hidden/HC chain is stable, wire gpu7 output-head logits and the selected
token oracle.
