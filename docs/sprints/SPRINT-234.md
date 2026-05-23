# Sprint 234 - Descriptor Backed One-Layer Execution

Date: 2026-05-23
Status: Complete

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

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Descriptor-backed mode reads the real `turbomind-pack-index.tsv`.
- [x] Descriptor-backed mode copies production-packed weight and scale bytes
      for the selected EP experts to the target GPUs.
- [x] TurboMind pointer tables are built from descriptor-derived byte offsets.
- [x] The smoke still opens the separate TP runtime at `32` slots / `256K`.
- [x] The smoke still verifies layer-2 ratio-4 KV with `max_abs=0`.
- [x] The smoke runs real TurboMind MXFP4 gated-SiLU/down kernels with
      descriptor-backed expert bytes on all eight GPUs.
- [x] V100 run passes finite deterministic repeat checks.
- [x] Evidence is copied to
      `logs/from-cluster/sprint234-tp-ep-descriptor-backed-layer/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack: `/workspace/packs/ds4-appliance-full-tm-gated-s181`

Command shape:

- `32` slots;
- `256K` context;
- `top_k=6`;
- layer `2`;
- MTP off;
- descriptor-backed experts enabled.

Descriptor-backed result:

| Metric | Value |
|---|---:|
| aggregate routes | `192` |
| dispatch bytes | `1572864` |
| return bytes | `1572864` |
| descriptor bytes read | `641728512` |
| runtime bytes/GPU | `7122628608` |
| KV max_abs | `0.000000000` |
| worst gate ms | `0.161109` |
| worst down ms | `0.085538` |
| worst EP ms | `0.246647` |
| dense/KV ms | `1.121624` |
| one-layer envelope ms | `1.368271` |
| repeat max_abs | `0.000000000` |
| repeat bad/nan | `0 / 0` |
| result | `PASS` |

Same-binary synthetic regression:

| Metric | Value |
|---|---:|
| worst EP ms | `0.247603` |
| dense/KV ms | `0.994061` |
| one-layer envelope ms | `1.241664` |
| repeat bad/nan | `0 / 0` |
| result | `PASS` |

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

Complete. The separate TP/EP layer smoke now has an opt-in
descriptor-backed expert mode that reads the real TurboMind index, copies
production-packed expert weight and scale bytes to each target V100, and feeds
descriptor-derived pointer tables into the real TurboMind MXFP4 gated-SiLU and
down kernels. This proves byte binding for the routed expert portion of layer
`2`.

This is still not TP/EP serving. Dense/control/router/attention descriptors
are not yet executed as real DS4 layer math, and logits equivalence is not
claimed. The next sprint should scale from descriptor-backed routed experts to
a descriptor-backed full-layer TP/EP decode gate with MTP still off.
