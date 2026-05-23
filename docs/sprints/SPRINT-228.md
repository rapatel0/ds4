# Sprint 228 - TP/EP Pack Contract

Date: 2026-05-23
Status: Complete - TP/EP Pack Contract Emitted

## Overview

Sprint 226 established the TP8/EP8 memory contract. Sprint 227 characterized
the TP8 collective boundary. The next step is to make the TP/EP pack ownership
model explicit so runtime work does not reinterpret the old PP/layer pack.

This sprint creates a TP/EP pack-contract generator. It reads the existing
production pack metadata as a source inventory, then emits a new TP/EP contract
manifest describing dense TP shards, EP expert ownership, sharded KV
descriptors, and per-GPU memory estimates.

This is still a contract sprint, not a byte-repacking sprint.

## Goals

- Add a standalone TP/EP pack contract tool.
- Read the current `pack-index.tsv` and `turbomind-pack-index.tsv` from a pack
  directory.
- Emit a TP/EP manifest with:
  - dense tensor TP shard descriptors;
  - replicated small control/router descriptors;
  - EP routed expert ownership by expert range;
  - sharded KV/cache descriptors for the DS4 compression schedule;
  - per-GPU memory summary.
- Default to the target shape:
  - TP=8;
  - EP=8;
  - slots=32;
  - ctx=256K;
  - KV dtype F8;
  - MTP off.
- Keep the tool separate from the PP packer and PP runtime.

## Non-Goals

- No actual TP/EP weight byte emission.
- No production runtime loader.
- No PP/layer-split pack format changes.
- No generic pack abstraction shared between PP and TP.
- No MTP pack contract.

## Implementation

1. Add `tools/ds4-v100-tp-ep-pack-contract.c`.
2. Add a Makefile target for the tool.
3. CLI:
   - `--pack-dir PATH`;
   - `--out-dir PATH`;
   - `--ctx N`;
   - `--slots N`;
   - `--kv-dtype f16|f8|q8_0`;
   - `--reserve-gib F`;
   - `--scratch-gib F`.
4. Emit:
   - `tp-ep-pack-contract.tsv`;
   - `tp-ep-memory-summary.tsv`;
   - `tp-ep-pack-contract.md`.
5. Contract rules:
   - routed expert tensors come from `turbomind-pack-index.tsv` and are split
     by expert id: `32` experts/GPU for EP8;
   - dense low-bit tensors are TP8 sharded by tensor dimension rule;
   - small F32/I32 control/router tensors are replicated unless the contract
     marks a specific future split rule;
   - KV rows use the DS4 corrected compression schedule:
     layers `0-1` SWA only, even `2..42` ratio `4` with indexer, odd `3..41`
     ratio `128`.
6. Validate locally with a synthetic minimal pack directory if no full pack is
   local.
7. Validate on the V100 pod against
   `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
8. Copy evidence to
   `logs/from-cluster/sprint228-tp-ep-pack-contract/`.
9. Update sprint/status/vision docs and commit.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-pack-contract.c` | TP/EP pack-contract generator |
| `Makefile` | build target |
| `docs/sprints/SPRINT-228.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint228-tp-ep-pack-contract/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Tool builds locally.
- [x] Tool builds/runs on the V100 pod.
- [x] Tool emits all three contract artifacts.
- [x] Manifest includes dense TP, replicated control, EP expert, and KV shard
      records.
- [x] Memory summary reports all eight GPUs and the target 32-slot/256K config.
- [x] Contract explicitly rejects PP/layer ownership.
- [x] Evidence is copied to
      `logs/from-cluster/sprint228-tp-ep-pack-contract/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
make -B -j8 tools/ds4-v100-tp-ep-pack-contract
git diff --check
synthetic minimal pack smoke:
  pack_rows=3 dense_rows=8 control_rows=8 expert_rows=8 kv_rows=840
```

V100 validation:

```text
cd /workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-tp-ep-pack-contract

./tools/ds4-v100-tp-ep-pack-contract \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --out-dir /workspace/logs/sprint228-tp-ep-pack-contract/contract \
  --ctx 262144 \
  --slots 32 \
  --kv-dtype f8 \
  --reserve-gib 2 \
  --scratch-gib 1.5
```

Generated artifacts:

```text
tp-ep-pack-contract.tsv       11121 lines
tp-ep-memory-summary.tsv          9 lines
tp-ep-pack-contract.md           35 lines
```

Record counts:

```text
pack_rows=1199
dense_rows=4096
control_rows=5496
expert_rows=688
kv_rows=840
```

Example manifest rows exist for each required class:

```text
dense_tp              token_embd.weight      split_axis=vocab
replicated_control    blk.0.attn_sinks       split_axis=replicate
ep_expert             blk.0.ffn_down_exps    expert_first=0 expert_count=32
kv_shard              kv.attn.blk.0          split_axis=kv_dim
kv_comp_state         kv.comp_state.blk.2    split_axis=state_dim
```

Per-GPU summary for the target `32` slot / `256K` / F8-KV contract:

```text
dense_tp:             1.006 GiB
replicated_control:   0.310 GiB
ep_expert:           17.133 GiB
kv:                   3.396 GiB
comp_state:           1.680 GiB
scratch:              1.500 GiB
reserve:              2.000 GiB
total:               27.024 GiB
```

Every GPU reports the same total because this contract is TP8/EP8 balanced:
`32` experts/GPU, dense shards split across all ranks, and KV sharded across
all ranks.

Evidence is stored in
`logs/from-cluster/sprint228-tp-ep-pack-contract/`.

## Decision

Sprint 228 ships the TP/EP pack contract generator. This does not emit new
weight shard bytes yet, but it makes the future TP/EP pack layout concrete:
dense tensors are TP-sharded, small F32/I32 control/router tensors are
replicated, routed experts are EP-sharded by expert id, and KV/cache ownership
is represented explicitly.

The generated real-pack contract stays inside the same memory envelope as
Sprint 226: `27.024 GiB` per GPU including reserve and scratch at
`32` slots / `256K`. This clears the pack-contract gate for the next sprint.

Next sprint should add the TP runtime skeleton that consumes this ownership
model, opens all eight GPUs, allocates resident hidden/KV/scratch arenas, and
executes no-op or fixture layer passes without touching the frozen PP
scheduler.
