# Sprint 226 - TP8/EP8 Planner Contract

Date: 2026-05-23
Status: Complete - TP8/EP8 Planner Contract Shipped

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

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] `tools/ds4-v100-plan-tp.c` no longer exposes PP/layer-split topology
      modes.
- [x] Default planner run describes only `TP8/EP8`, `PP=1`.
- [x] Planner reports configured `32` slots / `256K` memory with sharded
      quantized KV.
- [x] Planner reports context-tier admission for 128K, 256K, 512K, and 1M.
- [x] Planner reports hidden collective and routed EP traffic estimates.
- [x] Planner reports expert ownership and expected route density.
- [x] Local build and sample runs pass.
- [x] V100 build/run evidence is recorded.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
make -B -j8 tools/ds4-v100-plan-tp
./tools/ds4-v100-plan-tp --ctx 262144 --slots 32 --kv-dtype f8
./tools/ds4-v100-plan-tp --json
./tools/ds4-v100-plan-tp --topology tp4  # fails closed
git diff --check
```

V100 build and pack-derived planning:

```text
cd /workspace/ds4-sprint181
make -B -j80 tools/ds4-v100-plan-tp
./tools/ds4-v100-plan-tp \
  --ctx 262144 \
  --slots 32 \
  --kv-dtype f8 \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181
```

Cluster result with the real production pack bytes:

```text
topology: PP=1(no pipeline) TP=8 EP=8 KV=sharded
weights: total 145.42 GiB, per TP rank 18.18 GiB
verdict: fits; per-GPU total 27.00 / 32.00 GiB
headroom after reserve: 5.00 GiB

KV aggregate before TP sharding:
  attn_kv:             21.88 GiB aggregate, 2.73 GiB/GPU
  indexer_kv:           5.29 GiB aggregate, 0.66 GiB/GPU
  comp_state envelope: 13.44 GiB aggregate, 1.68 GiB/GPU

Admission tiers:
  128K: 126 slots
  256K:  63 slots
  512K:  31 slots
  1M:    15 slots

Decode-shape traffic:
  hidden payload per rank:          0.250 MiB
  one ring all-reduce per rank:     0.438 MiB
  hidden collectives, 2/layer x 43: 37.625 MiB
  EP dispatch + return aggregate:   3.000 MiB

Expert density:
  experts per GPU: 32
  active routes per decode step: 192
  average routes per GPU: 24.00
  average routes per expert: 0.750
```

Negative guard:

```text
./tools/ds4-v100-plan-tp --topology tp4
ds4-v100-plan-tp: --topology was removed; this planner is TP8/EP8-only
```

Evidence is stored in
`logs/from-cluster/sprint226-tp-ep-planner/`.

## Decision

Sprint 226 ships the TP8/EP8 planner contract and removes the old PP-style
topology modes from `tools/ds4-v100-plan-tp.c`.

The real-pack result keeps the hard-cut TP/EP direction viable: the target
`32` slots / `256K` context shape fits under the 32 GiB V100 budget with
sharded F8 KV and a 2 GiB reserve. The planner estimates 63 admitted slots at
256K and 31 at 512K with the current assumptions; 32 slots at 512K is just over
the conservative budget and should not be treated as admitted until measured
allocator overhead and compression-state sizing are tightened.

The next sprint should build the TP8 collective workbench against this exact
contract: 32 active slots first, then 64 and 128 token payloads as density
controls, with hidden reductions and EP dispatch/reduction traffic measured
inside the TP-only path.
