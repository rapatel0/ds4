# Sprint 086 Report: TurboMind Sidecar VRAM Admission

## Outcome

`SHIP_SIDECAR_ADMISSION`.

Sprint 086 added a CPU memory admission tool for TurboMind sidecars. The tool
reads the normal DS4 V100 pack index and the derived TurboMind sidecar index,
then reports per-GPU source arena bytes, source expert payload bytes, sidecar
bytes, duplicate-residency totals, and replacement-style totals.

This directly addresses the 32 GB V100 constraint: full TurboMind expert packs
must be explicitly admitted and should generally replace source expert
residency rather than silently duplicating it.

## What Changed

- Added `tools/ds4-v100-turbomind-admit.c`.
- Added Makefile build/clean rules for `tools/ds4-v100-turbomind-admit`.
- Recorded cluster validation in
  `logs/from-cluster/sprint086-turbomind-admit-v100.log`.

## V100 Evidence

Build and run:

```sh
make tools/ds4-v100-turbomind-admit

./tools/ds4-v100-turbomind-admit \
  --source-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --tm-index /tmp/ds4-sprint085-tm-pack/turbomind-pack-index.tsv \
  --gpus 8 \
  --vram-gib 32 \
  --reserve-gib 4 \
  --kv-gib 1 \
  --scratch-gib 1
```

Result summary:

```text
summary duplicate_fit=yes replacement_fit=yes vram_gib=32.000 reserve_gib=4.000 kv_gib=1.000 scratch_gib=1.000
```

The bounded sidecar only adds `0.025 GiB` on GPU 0, so duplicate residency is
still fine for this smoke. The important production signal is the source expert
payload already resident per GPU:

```text
gpu0 source_expert_payload_gib=19.125
gpu1 source_expert_payload_gib=19.125
gpu2 source_expert_payload_gib=19.125
gpu3 source_expert_payload_gib=19.125
gpu4 source_expert_payload_gib=19.125
gpu5 source_expert_payload_gib=15.938
gpu6 source_expert_payload_gib=15.938
gpu7 source_expert_payload_gib=9.562
```

## Decision

Do not make full TurboMind sidecars an unaccounted duplicate cache. The
production direction should either:

- replace source expert payload with TurboMind packed experts in the runtime
  pack, or
- admit only a bounded sidecar cache whose bytes are included in the planner.

## Risks

- Replacement-style totals are payload estimates, not a compacted shard format.
- A full sidecar still needs actual generation and validation, because packed
  bytes should be measured from the TurboMind ABI rather than guessed.
- Scheduler integration is still pending.
