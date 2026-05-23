# Sprint 226 - TP8/EP8 Planner Contract

Date: 2026-05-23
Status: Planned

## Overview

The vision now makes a hard cut away from PP/layer-split optimization. The
current `tools/ds4-v100-plan-tp.c` still compares PP-style topology variants
such as `PP8/TP1`, `PP4/TP2`, and `PP2/TP4`. That was useful during
investigation, but it is now the wrong contract: future TP/EP runtime work
should not be able to inherit layer-split assumptions by accident.

Sprint 226 replaces/refocuses the TP planner as a TP8/EP8-only planning tool.
It should answer whether the intended appliance shape fits and what the runtime
must allocate, shard, and communicate.

Target:

```text
8x V100-SXM2-32GB
pipeline parallel = 1 (no pipeline stages)
tensor parallel   = 8
expert parallel   = 8
slots             = 32
context           = 256K minimum
KV                = sharded, quantized by default
MTP               = off
```

## Goals

- Replace the legacy multi-topology TP planner output with a single TP8/EP8
  topology contract.
- Report memory by GPU for:
  - source/resident weight estimate;
  - sharded DS4 compressed KV;
  - compression/indexer state;
  - scratch;
  - collective workspace;
  - reserve and headroom.
- Report admission by context tier for 128K, 256K, 512K, and 1M.
- Report decode-shape traffic estimates for hidden collectives and EP routed
  dispatch/return at the configured slot count.
- Report expert ownership and expected route density for `32` active slots.
- Keep this as a separate TP/EP codepath. Do not add generic scheduler
  abstractions and do not extend PP/layer-split topology planning.

## Non-Goals

- No PP/layer-split variants.
- No `TP4` fallback/control topology in this planner.
- No production runtime changes.
- No MTP enablement.
- No throughput claim from this sprint. This is a topology and memory contract.
- No replicated-KV production path. Replicated KV may be printed only as a
  labeled anti-goal/sensitivity check if useful.

## Implementation

1. Refactor `tools/ds4-v100-plan-tp.c` into a TP8/EP8-only planner.
2. Keep the planner standalone instead of sharing planner abstractions with the
   frozen PP path.
3. Add CLI options for:
   - `--ctx`;
   - `--slots`;
   - `--kv-dtype f16|f8|q8_0`;
   - `--weight-total-gib` for static estimation;
   - `--pack-dir` to derive total resident source bytes from an existing pack
     when running on the cluster;
   - `--reserve-gib`;
   - `--scratch-gib`;
   - `--json`.
4. Use DS4 corrected KV layout:
   - layers `0-1`: SWA-only, `128` rows;
   - even layers `2..42`: ratio `4`, with indexer KV;
   - odd layers `3..41`: ratio `128`, no indexer KV;
   - `attn_kv` dim `512`;
   - `indexer_kv` dim `128`.
5. Shard KV across TP8 by default and fail closed if a non-sharded production
   mode is requested.
6. Print route density at `slots * top_k`, with `top_k=6` and `256` experts
   owned as `32` experts per GPU.
7. Build locally and run representative planner commands.
8. Build/run on the V100 pod if cluster access is available, using the existing
   production pack directory as a weight-size source estimate.
9. Record output in `logs/from-cluster/sprint226-tp-ep-planner/` when cluster
   validation runs.
10. Update this sprint document with evidence and decision.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-plan-tp.c` | TP8/EP8-only memory/topology planner |
| `docs/sprints/SPRINT-226.md` | plan and execution evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update if the planner contract changes the sequence |
| `logs/from-cluster/sprint226-tp-ep-planner/` | cluster evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] `tools/ds4-v100-plan-tp.c` no longer exposes PP/layer-split topology
      modes.
- [ ] Default planner run describes only `TP8/EP8`, `PP=1`.
- [ ] Planner reports configured `32` slots / `256K` memory with sharded
      quantized KV.
- [ ] Planner reports context-tier admission for 128K, 256K, 512K, and 1M.
- [ ] Planner reports hidden collective and routed EP traffic estimates.
- [ ] Planner reports expert ownership and expected route density.
- [ ] Local build and sample runs pass.
- [ ] V100 build/run evidence is recorded if the pod is reachable.
- [ ] Changes are committed with explicit `git add` paths.

## Decision

Pending.
