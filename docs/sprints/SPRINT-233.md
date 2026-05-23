# Sprint 233 - Descriptor Driven TP/EP Layer Gate

Date: 2026-05-23
Status: Planned

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

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Descriptor parsing is TP/EP-only and does not modify PP scheduler files.
- [ ] The smoke reads a real TP/EP pack contract or equivalent real-pack view.
- [ ] Layer `2` descriptor view reports dense, control/router, EP expert, and
      KV descriptor counts.
- [ ] The smoke validates TP8/EP8/PP1 ownership and sharded KV.
- [ ] The smoke reports per-GPU expert ownership and local expert counts.
- [ ] Unsupported descriptor families fail closed with a useful message.
- [ ] V100 build/run passes against the production pack contract.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint233-tp-ep-descriptor-layer/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

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

Pending.
