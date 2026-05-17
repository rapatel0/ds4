---
sprint: 004
title: Runtime Pack Loading And V100 Device Residency Smoke
status: completed
date: 2026-05-17
verdict: SHIP
---

# SPRINT-004 Report

## Verdict

`SHIP`.

Sprint 004 proved the structural residency contract for the DS4 V100
appliance:

- runtime pack-index parsing and source reconciliation are implemented;
- source GGUF reconciliation passed for the real model;
- full `gpuN.weights` shards were emitted to persistent scratch;
- both GGUF and shard providers loaded all packed bytes into 8 V100 device
  arenas;
- all spot checks and the deterministic provider cross-check passed;
- source-model generation remains guarded.

## Implementation Summary

### Phase 0: Orientation And Build Hygiene

Completed.

- Weather report fetched; implementation model guidance points at
  `gpt-5.3-codex`.
- Local `make cpu` passed.
- Local `make tools/ds4-v100-pack` passed.
- Cluster procedure from
  `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`
  was reviewed.
- Dedicated pod `ds4-sprint004-8gpu` was created on `gpu-01`.
- Persistent scratch was mounted at `/workspace`, backed by
  `/srv/dev/ds4-sprint004`.
- Scratch check: `/workspace` had 257 GiB free before full shard emission.
- CUDA environment: CUDA 12.2 image, driver `580.126.20`, 8x
  Tesla V100-SXM2-32GB visible.

### Phase 1: Pack-Index Reader And Reconciliation

Completed.

Implemented:

- `ds4_pack.h`
- `ds4_pack.c`
- `--pack-index`
- `--pack-reconcile-report`
- inspect-time pack reconciliation in `ds4_engine_open`
- `tests/pack_index_smoke.c`

Real-model reconciliation:

```text
source_tensors=1328
pack_rows=1328
ok=1328
```

Artifact:

- `docs/sprints/drafts/SPRINT-004-RECONCILE.log`

### Phase 2: Upload-Only Per-GPU Arena Sidecar

Completed.

Implemented:

- residency-only `ds4_gpu_arena_*` API in `ds4_gpu.h`
- CUDA arena implementation in `ds4_cuda.cu`
- CPU/stub arena implementation in `ds4_gpu_arena_stub.c`
- `tests/gpu_arena_smoke.c`

CUDA success paths use `cudaMalloc` device memory and report memory kind as
`device`. CPU/stub paths report `host-stub` and do not claim real residency.

### Phase 3: Local Synthetic Residency Smoke

Completed.

Implemented:

- `tools/ds4-v100-residency-smoke.c`
- `tests/residency_smoke_synthetic.sh`

Validated locally and in the V100 pod:

```text
pack_index_smoke: ok
gpu_arena_smoke: ok
residency_smoke_synthetic: ok
```

### Phase 4: Cluster Shard Emission On Persistent Scratch

Completed.

Full real-model shard emission succeeded under `/workspace/ds4-pack`.

Shard sizes:

```text
gpu0.weights  22524134668
gpu1.weights  21494393612
gpu2.weights  21494393612
gpu3.weights  21494393612
gpu4.weights  21494393612
gpu5.weights  17922654732
gpu6.weights  17901334540
gpu7.weights  11817197824
```

Artifacts:

- `docs/sprints/drafts/SPRINT-004-PACK-EMIT.log`
- `docs/sprints/drafts/SPRINT-004-SHARD-SIZES.tsv`
- `docs/sprints/drafts/SPRINT-004-SHARD-SHA256.tsv`

### Phase 5: Cluster Per-GPU Residency Smoke

Completed.

Both providers passed on 8 V100s.

Common result:

```text
pack_rows       1328
uploaded_tensors 1328
uploaded_bytes 156142862684
spot_checks    2230
visible_devices 8
required_devices 8
memory_kind    device
result         OK
```

Per-GPU arena bytes:

| GPU | Arena Bytes | Free After Alloc/Upload |
|---:|---:|---:|
| 0 | 22524134668 | 11220287488 |
| 1 | 21494393612 | 12249989120 |
| 2 | 21494393612 | 12249989120 |
| 3 | 21494393612 | 12249989120 |
| 4 | 21494393612 | 12249989120 |
| 5 | 17922654732 | 15821438976 |
| 6 | 17901334540 | 15842410496 |
| 7 | 11817197824 | 21928345600 |

The tightest GPU remains above the 3 GiB reserve. All 8 GPUs reported
P2P-visible access to all peers.

Cross-provider check:

```text
crosscheck blk.0.attn_sinks 256 OK
```

Artifacts:

- `docs/sprints/drafts/SPRINT-004-RESIDENCY-GGUF.log`
- `docs/sprints/drafts/SPRINT-004-RESIDENCY-SHARD.log`
- `docs/sprints/drafts/SPRINT-004-CROSSCHECK.log`

### Phase 6: Report And Follow-Ups

Completed.

Follow-ups were written to:

- `docs/sprints/SPRINT-004-FOLLOWUPS.md`

## Validation

Local:

```text
make cpu
make tools/ds4-v100-pack
make tools/ds4-v100-residency-smoke
make tests/pack_index_smoke
make tests/gpu_arena_smoke
./tests/pack_index_smoke
./tests/gpu_arena_smoke
tests/residency_smoke_synthetic.sh
git diff --check
```

Cluster:

```text
make cpu
make tools/ds4-v100-pack
make tools/ds4-v100-residency-smoke CUDA_ARCH=sm_70
make tests/pack_index_smoke tests/gpu_arena_smoke
./tests/pack_index_smoke
./tests/gpu_arena_smoke
tests/residency_smoke_synthetic.sh
./ds4 --inspect --cpu -m /models/DSv4-Flash-256e-fixed.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --pack-reconcile-report /workspace/ds4/SPRINT-004-RECONCILE.log
./tools/ds4-v100-pack --manifest docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /workspace/ds4-pack.tmp --gpus 8 --write-index --emit-shards
./tools/ds4-v100-residency-smoke --provider gguf ...
./tools/ds4-v100-residency-smoke --provider shard --crosscheck ...
```

`make test` on the laptop was not counted as a Sprint 004 gate because it
fails before the new code path due to the missing default `ds4flash.gguf`
model file.

## Architecture Deltas

No material changes are required to `docs/architecture/DS4-V100-LAYOUT.md`.
The observed per-GPU resident weight bytes match the Sprint 003 planner, and
the 8x V100 32 GiB memory budget is viable for packed weight residency.

## Guard Status

The source-model generation guard remains active. Sprint 004 added
inspect/reconciliation and residency smoke paths only; it did not enable
source-model decode, MTP, speculative decoding, or source-format math.

Artifact:

- `docs/sprints/drafts/SPRINT-004-GUARD.log`
