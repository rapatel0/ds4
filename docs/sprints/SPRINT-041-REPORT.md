# Sprint 041 Report: MTP Rollback State Safety

## Outcome

`SHIP`.

Sprint 041 added target scheduler snapshot/restore APIs, a deeper scheduler
snapshot smoke, and an MTP rollback gate. It does not claim native
prompt-token MTP verify. The full V100 gate passes with zero failures and now
honestly reports:

```text
gate	readiness	NOT_READY	missing=mtp_verify
gate	summary	PASS	failures=0 ready=false
```

## Implementation

- Added `ds4_v100_stage_scheduler_snapshot` create/restore/free APIs.
- Snapshot coverage includes current HC buffer identity/content, raw KV,
  attention compressor state/cache/counters, indexer compressor
  state/cache/counters, and indexer top-k tensors.
- Added `tests/cuda_v100_scheduler_snapshot_smoke.c`.
- Added `tools/ds4-v100-mtp-verify-smoke.c`, wired as gate label
  `mtp_rollback`.
- Wired `scheduler_snapshot` and `mtp_rollback` into
  `tools/ds4-v100-gate.sh`.

## Validation

Local object compile:

```text
make tools/ds4-v100-mtp-verify-smoke.o tests/cuda_v100_scheduler_snapshot_smoke.o
```

Focused V100 build:

```text
CUDA_ARCH=sm_70 make tests/cuda_v100_scheduler_snapshot_smoke tools/ds4-v100-mtp-verify-smoke
```

Focused scheduler snapshot smoke on 8x V100:

```text
cuda_v100_scheduler_snapshot_smoke: token=16 steps=8 next=17 snapshot_bytes=30064724 before_top1=17 restored_top1=17 replay_top1=24 hc_mutate_delta=68.5005646 hc_restore_delta=0 restore_delta=0 replay_delta=0 PASS
```

Focused MTP rollback smoke on 8x V100:

```text
mtp_rollback_smoke: prompt_tokens=18 committed=926 target_top1=1 rejected=16 snapshot_bytes=30107648 restore_delta=0 replay_delta=0 mtp_raw_max_abs=0 PASS
```

Full V100 gate:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build --model /models/DSv4-Flash-256e-fixed.gguf --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --ctx 1048576 --slots 1 --log-dir docs/sprints/drafts/SPRINT-041-GATE-CLUSTER-8GPU
```

Result: all gates passed; readiness remains blocked on native `mtp_verify`.

## Evidence

- Gate logs: `docs/sprints/drafts/SPRINT-041-GATE-CLUSTER-8GPU/`
- Focused rollback report:
  `docs/sprints/drafts/SPRINT-041-MTP-ROLLBACK/mtp_rollback.report`
- Full-gate rollback report:
  `docs/sprints/drafts/SPRINT-041-GATE-CLUSTER-8GPU/mtp_rollback.report`

Key MTP residency numbers from the gate rollback report:

- sidecar arena bytes: `3807601408`
- sidecar uploaded bytes: `3807600108`
- gpu7 free after sidecar upload: `29937369088`
- reserve bytes: `4294967296`

## Remaining Blocker

Native prompt-token `mtp_verify` is still missing. Sprint 042 should connect
the resident MTP forward path to the actual just-committed token embedding and
target HC state, produce a real draft token, compare it against target top-1,
and then prove accept/reject state transitions.
