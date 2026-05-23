# Sprint 228 - TP/EP Pack Contract

Date: 2026-05-23
Status: Planned

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

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Tool builds locally.
- [ ] Tool builds/runs on the V100 pod.
- [ ] Tool emits all three contract artifacts.
- [ ] Manifest includes dense TP, replicated control, EP expert, and KV shard
      records.
- [ ] Memory summary reports all eight GPUs and the target 32-slot/256K config.
- [ ] Contract explicitly rejects PP/layer ownership.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint228-tp-ep-pack-contract/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Decision

Pending.
