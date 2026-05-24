---
sprint: 327
title: TP/EP Production Compressed-KV Memory Contract
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 327 - TP/EP Production Compressed-KV Memory Contract

## Goal

Turn the Sprint 326 bounded compressed-row diagnostic into an explicit
production memory contract for TP/EP KV ownership, so the next runtime allocator
can be implemented against concrete row and byte budgets instead of growing
diagnostic buffers until VRAM fails.

## Why This Sprint

Sprint 326 proved bounded multi-row compressed attention behavior at `32` slots
and `256K`, but it also exposed the remaining production gap: diagnostic f32
state and per-layer helper buffers can sit near the V100 32GB limit, while the
serving runtime needs a compact, typed, sharded KV allocator.

Before changing the serving allocator, the TP planner should state exactly:

- raw SWA rows per layer
- compressed attention rows per ratio schedule
- indexer rows on ratio-4 layers
- bytes by dtype and sharding mode
- current diagnostic replicated f32 pressure
- admission limits for `128K`, `256K`, `512K`, and `1M`

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint is a planning/contract
and validation sprint, but it must produce executable tooling and cluster
artifacts, not just prose.

## Implementation Plan

1. Extend `tools/ds4-v100-plan-tp.c` with a compressed-KV contract section:

- per-layer ratio
- raw SWA rows
- compressed attention rows
- indexer rows
- aggregate persistent values
- per-GPU bytes for the selected KV dtype
- diagnostic f32 replicated-buffer bytes

2. Add machine-readable JSON fields for the same contract.

3. Run the planner against:

- `32` slots / `256K` / F8 using the real TP pack directory
- `1` slot / `1M` / F8 using the real TP pack directory

4. Record the outputs as cluster artifacts and copy them locally.

## Definition of Done

- [x] `tools/ds4-v100-plan-tp.c` builds locally or on the V100 pod.
- [x] The planner prints a compressed-KV contract in human output.
- [x] JSON output exposes compressed-KV row and byte totals.
- [x] `32` slot / `256K` / F8 with the real pack reports whether it fits.
- [x] `1` slot / `1M` / F8 with the real pack reports whether it fits.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_039.md`, cluster/local artifacts, and
  this sprint doc are updated and committed.

## Outcome

Extended `tools/ds4-v100-plan-tp.c` with a production compressed-KV contract:

- raw SWA rows across all layers
- ratio-4 attention compressed rows
- ratio-128 attention compressed rows
- ratio-4 indexer compressed rows
- persistent KV values and bytes for the selected KV dtype
- per-TP-rank persistent KV bytes
- replicated f32 bytes per GPU as a non-target warning
- current bounded diagnostic f32 bytes per GPU
- per-layer row and byte table
- JSON fields for the same contract

This establishes the runtime allocator contract numerically: production serving
must use typed, TP-sharded KV. Replicated f32 KV is not viable for the target
32-slot / 256K configuration.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 tools/ds4-v100-plan-tp
```

Result: PASS.

Real-pack planner runs:

| Case | Verdict | Per-GPU total | Headroom after reserve | Persistent KV / GPU | Replicated f32 / GPU |
|---|---:|---:|---:|---:|---:|
| `slots=32`, `ctx=262144`, `kv=f8` | fits | `27.00 GiB` | `5.00 GiB` | `3.40 GiB` | `107.84 GiB` |
| `slots=1`, `ctx=1048576`, `kv=f8` | fits | `22.56 GiB` | `9.44 GiB` | `0.42 GiB` | `13.45 GiB` |

For the target `32` slot / `256K` case, the row contract is:

- raw SWA rows, all layers: `5504`
- ratio-4 attention compressed rows: `1376256`
- ratio-128 attention compressed rows: `40960`
- ratio-4 indexer compressed rows: `1376256`
- persistent KV bytes, aggregate F8: `27.17 GiB`
- persistent KV bytes, per TP rank: `3.40 GiB`
- current bounded diagnostic f32 per GPU: `1.65 GiB`

Artifacts:

- `logs/from-cluster/sprint327-tp-kv-contract/cluster/plan-slots32-ctx262144-f8.md`
- `logs/from-cluster/sprint327-tp-kv-contract/cluster/plan-slots1-ctx1048576-f8.md`
- `logs/from-cluster/sprint327-tp-kv-contract/cluster/plan-slots32-ctx262144-f8.json`

## Next Step

Implement the runtime allocator against this contract:

- allocate typed TP-sharded persistent compressed rows instead of f32 bounded
  diagnostic rows
- keep layer-local active aliases
- stage only the rows required by the local attention read
- validate ratio-4 and ratio-128 history from the production arena
- rerun HTTP parity after attention output has full reference coverage

## Risks

- The planner can only encode the intended production allocator; it does not
  itself make serving use that allocator.
- Current runtime diagnostics still use f32 helper buffers and are allowed to
  exceed the production contract. The output must distinguish these clearly.
- If pack weights already consume too much VRAM for the desired slot/context
  target, this sprint should surface that numerically rather than hiding it
  behind optimistic assumptions.
