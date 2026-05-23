# Sprint 233 - Descriptor Driven TP/EP Layer Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 232 proved the combined TP runtime plus EP expert fixture in one
process at the target `32` slot / `256K` shape. Sprint 233 replaces synthetic
fixture ownership with descriptor-driven layer metadata from the real
production pack contract.

This is still a one-layer gate. It should not become a serving integration and
should not touch the frozen PP scheduler. The purpose is to make the TP/EP
runtime consume real pack descriptors and prove that a single DS4 layer can be
mapped into the new ownership model before scaling to all 43 layers.

## Goals

- Add a TP/EP layer descriptor loader for the separate TP/EP path.
- Read the Sprint 228 TP/EP pack contract output or generate the equivalent
  descriptor view from the production pack.
- Select one representative ratio-4 layer, default layer `2`.
- Resolve for that layer:
  - dense TP descriptor rows;
  - replicated control/router descriptor rows;
  - EP expert descriptor rows;
  - KV descriptor rows.
- Validate that descriptor ownership matches the TP/EP contract:
  - `TP=8`;
  - `EP=8`;
  - `PP=1`;
  - `32` local experts per rank;
  - sharded KV, not replicated KV.
- Feed descriptor-derived EP expert ownership into a one-layer smoke.
- Keep the current synthetic hidden/routes if required, but expert ownership
  and tensor byte spans must come from descriptors, not hard-coded constants.
- Report missing or unsupported descriptor families explicitly.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No full 43-layer decode.
- No serving integration.
- No MTP.
- No logits equivalence claim unless the descriptor-backed layer actually
  executes all required real math.

## Implementation

1. Add a TP/EP descriptor loader module or tool-local parser.
2. Prefer a new file over extending PP context abstractions.
3. Add a smoke mode to `tools/ds4-v100-tp-ep-layer-smoke.cu` or a new
   companion tool if cleaner:
   - `--pack-contract PATH`;
   - `--layer N`;
   - `--require-dense`;
   - `--require-experts`;
   - `--require-kv`.
4. Parse the contract rows and build a per-rank layer view.
5. Validate counts and byte spans for layer `2`.
6. If safe within this sprint, bind descriptor-derived active experts into the
   existing Sprint 232 EP smoke path.
7. Build on the V100 pod.
8. Run against the real contract generated from
   `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
9. Copy evidence to
   `logs/from-cluster/sprint233-tp-ep-descriptor-layer/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-layer-smoke.cu` | descriptor-backed mode, if reused |
| `tools/ds4-v100-tp-ep-layer-descriptor-smoke.c` or `.cu` | companion tool, if cleaner |
| `Makefile` | build target if a new tool is added |
| `docs/sprints/SPRINT-233.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome/sequence update |
| `logs/from-cluster/sprint233-tp-ep-descriptor-layer/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Descriptor parsing is TP/EP-only and does not modify PP scheduler files.
- [x] The smoke reads a real TP/EP pack contract or equivalent real-pack view.
- [x] Layer `2` descriptor view reports dense, control/router, EP expert, and
      KV descriptor counts.
- [x] The smoke validates TP8/EP8/PP1 ownership and sharded KV.
- [x] The smoke reports per-GPU expert ownership and local expert counts.
- [x] Unsupported descriptor families fail closed with a useful message.
- [x] V100 build/run passes against the production pack contract.
- [x] Evidence is copied to
      `logs/from-cluster/sprint233-tp-ep-descriptor-layer/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

Build and run on V100:

```text
cd /workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-tp-ep-layer-descriptor-smoke
./tools/ds4-v100-tp-ep-layer-descriptor-smoke \
  --contract /workspace/logs/sprint228-tp-ep-pack-contract/contract/tp-ep-pack-contract.tsv \
  --layer 2 --samples 8
```

Result:

```text
layer_descriptor_summary layer 2 total_rows 288 dense_rows 112 control_rows 136 expert_rows 16 kv_rows 16 comp_rows 8 bad_rows 0
gpu 0 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 0 expert_count 32 mismatches 0
gpu 1 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 32 expert_count 32 mismatches 0
gpu 2 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 64 expert_count 32 mismatches 0
gpu 3 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 96 expert_count 32 mismatches 0
gpu 4 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 128 expert_count 32 mismatches 0
gpu 5 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 160 expert_count 32 mismatches 0
gpu 6 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 192 expert_count 32 mismatches 0
gpu 7 rows 36 bytes 711945176 dense_rows 14 control_rows 17 expert_rows 2 kv_rows 2 comp_rows 1 expert_first 224 expert_count 32 mismatches 0
tp_ep_layer_descriptor_smoke layer 2 pp 1 tp 8 ep 8 global_experts 256 local_experts 32 kv sharded PASS
```

Logs:

- `logs/from-cluster/sprint233-tp-ep-descriptor-layer/layer2-descriptor-smoke.log`

## Risks

- The Sprint 228 contract is a metadata contract, not repacked TP/EP bytes. If
  it lacks enough offsets to execute real weights directly, this sprint should
  stop at descriptor validation plus descriptor-derived ownership and record
  the missing byte-level requirement for the next sprint.
- Real layer execution may require dense/router descriptors that are not yet
  represented in a runtime-friendly format. That should not trigger PP
  abstraction work.
- A parser bug here could give false confidence. The smoke should print raw
  counts and sample rows in evidence.

## Decision

Sprint 233 passes as a descriptor metadata and ownership gate. Layer `2` in the
real TP/EP contract resolves cleanly to TP8/EP8/PP1 with dense TP rows,
replicated control/router rows, EP expert rows, sharded KV rows, and compression
state rows on every GPU. Per-GPU ownership is balanced at `36` rows and
`711945176` estimated bytes for this layer, with expert ranges
`0..31`, `32..63`, ..., `224..255` and zero ownership mismatches.

This sprint does not execute real descriptor-backed weights. The next sprint
must bind these descriptor rows to actual byte spans in the production pack and
feed descriptor-derived expert pointers into the one-layer TP/EP smoke.
