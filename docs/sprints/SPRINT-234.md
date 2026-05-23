# Sprint 234 - Descriptor Backed One-Layer Execution

Date: 2026-05-23
Status: Planned

## Overview

Sprint 233 proved that layer `2` in the real production TP/EP contract has the
right ownership metadata. Sprint 234 binds the routed-expert portion of that
layer to actual production-pack TurboMind bytes and runs the one-layer TP/EP
smoke with descriptor-derived expert pointers.

This is still not serving and not a full logits-equivalent DS4 layer. It is the
byte-binding gate between metadata validation and real layer execution.

## Goals

- Add descriptor-backed mode to the TP/EP one-layer smoke.
- Read the production `turbomind-pack-index.tsv`.
- Resolve layer `2` routed expert entries:
  - `blk.2.ffn_gate_up_exps.weight`;
  - `blk.2.ffn_down_exps.weight`.
- Open the sidecar pack file referenced by the TurboMind index.
- For each EP rank, select local active experts from the descriptor ownership
  range:
  - GPU0 uses global experts `0..5`;
  - GPU1 uses `32..37`;
  - ...
  - GPU7 uses `224..229`.
- Copy production-packed weight and scale bytes for those experts to the owning
  target GPU.
- Build the TurboMind pointer tables from descriptor-derived offsets, not
  synthetic fixture packing.
- Run the one-layer TP/EP smoke at:
  - `32` slots;
  - `256K` context;
  - `top_k=6`;
  - layer `2`;
  - MTP off.
- Validate finite deterministic repeat output, KV update correctness, and
  per-rank timing.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No dense descriptor-backed execution yet.
- No real router/top-k computation yet.
- No logits equivalence claim.
- No serving integration.
- No MTP.

## Implementation

1. Extend `tools/ds4-v100-tp-ep-layer-smoke.cu` with:
   - `--tm-index PATH`;
   - `--pack-dir DIR`;
   - `--descriptor-backed-experts`.
2. Add a tool-local TurboMind index parser for the two layer-2 expert rows.
3. Add a packed-byte loader that reads:
   - `weight_offset + global_expert * weight_bytes_per_expert`;
   - `scale_offset + global_expert * scale_bytes_per_expert`.
4. Upload those bytes to the target rank and fill `StridedPtrH` pointer tables.
5. Preserve the synthetic-hidden route distribution from Sprint 232, so only
   expert bytes change in this sprint.
6. Keep synthetic fixture mode available as a diagnostic fallback.
7. Build and run on the V100 pod against
   `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
8. Copy evidence to
   `logs/from-cluster/sprint234-tp-ep-descriptor-backed-layer/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-layer-smoke.cu` | descriptor-backed expert mode |
| `docs/sprints/SPRINT-234.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint234-tp-ep-descriptor-backed-layer/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Descriptor-backed mode reads the real `turbomind-pack-index.tsv`.
- [ ] Descriptor-backed mode copies production-packed weight and scale bytes
      for the selected EP experts to the target GPUs.
- [ ] TurboMind pointer tables are built from descriptor-derived byte offsets.
- [ ] The smoke still opens the separate TP runtime at `32` slots / `256K`.
- [ ] The smoke still verifies layer-2 ratio-4 KV with `max_abs=0`.
- [ ] The smoke runs real TurboMind MXFP4 gated-SiLU/down kernels with
      descriptor-backed expert bytes on all eight GPUs.
- [ ] V100 run passes finite deterministic repeat checks.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint234-tp-ep-descriptor-backed-layer/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- The production pack uses source files according to the current PP appliance
  layout, so the layer-2 expert rows may all source from `gpu0.weights`.
  That is acceptable for this byte-binding gate because the TP/EP runtime is
  allowed to repack or redistribute bytes in later sprints.
- This does not make dense, router, or attention descriptor-backed yet.
- If the TurboMind packed bytes expose NaNs with synthetic activations, reduce
  input amplitude and record the numeric range issue rather than changing the
  ownership model.

## Decision

Pending.
