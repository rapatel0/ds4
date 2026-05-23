# Sprint 239 - TP/EP Layer-2 Next-Hidden Composition

Date: 2026-05-23
Status: Planned

## Overview

Sprint 238 closed the layer-2 dense coverage gap for the separate TP/EP path:
F8 dense groups, BF16 compressor/indexer groups, sharded KV allocation/update,
and EP TurboMind experts all execute from production bytes. The remaining gap
before serving is dataflow composition. Sprint 239 turns the full-layer smoke
from independent checks into a representative layer-2 next-hidden composition.

This sprint remains TP/EP-only. It does not touch the frozen PP scheduler and
does not introduce a generic scheduler abstraction.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in TP/EP-only `--compose-next-hidden` mode to the separate
  full-layer smoke.
- Build a route-to-slot map matching the existing EP route schedule.
- Run routed expert gate/up and down through the real TurboMind MXFP4 kernels.
- Reduce routed expert outputs by slot into hidden shards.
- Return expert contributions across GPUs with explicit peer copies.
- Compose per-rank next-hidden shards from:
  - `blk.2.attn_output_b.weight` F8 dense output;
  - `blk.2.ffn_down_shexp.weight` F8 shared-FFN output;
  - returned EP expert contributions;
  - residual deterministic input.
- Keep the sharded KV slice update/check in the same run.
- Report per-rank next-hidden checksum, finite checks, contribution bytes,
  peer return bytes, composition time, and final pass/fail.
- Run and record evidence on the V100 pod at `32` slots / `256K`, MTP off.

## Non-Goals

- No PP scheduler edits.
- No changes to `ds4_v100_scheduler.*`.
- No logits equivalence claim.
- No full attention softmax over raw plus compressed KV.
- No production server integration.
- No MTP.
- No HMMA/CUTLASS dense optimization yet.
- No claim that the representative composition exactly matches DS4 layer-2
  numerics. The goal is resident TP/EP dataflow and correctness of the
  movement/reduction mechanics.

## Architecture

Extend only the separate TP/EP full-layer smoke:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
  --dense-compute-all              # Sprint 238 coverage
  --compose-next-hidden            # Sprint 239 dataflow composition
```

The route schedule already assigns `slots * top_k` routes across EP ranks. The
new mode must preserve that schedule but also materialize `route_slot` on each
rank. After TurboMind down projection produces full hidden vectors for local
expert routes, the tool reduces those vectors into eight destination hidden
shards:

```text
source EP rank p:
  d_down[routes, 4096] half
  route_slot[routes] int
  ep_contrib[dest_rank, slots, 512] float

for dest rank d:
  ep_contrib[d, route_slot, hidden[d*512:(d+1)*512]] += d_down[route]
```

Then each source rank peer-copies its contribution shard to the destination
rank. Each destination rank composes:

```text
next_hidden_shard =
  residual_shard +
  attn_output_b_shard +
  shared_ffn_down_shard +
  sum(ep_contrib_from_all_sources)
```

This keeps memory movement explicit and measurable:

```text
ep_return_bytes = 8 sources * 8 destinations * slots * 512 * sizeof(float)
```

At `32` slots this is `4194304` bytes for the representative float return
path. A later optimized runtime can use half return or fused reduce/copy once
the dataflow is correct.

## Implementation

1. Add `--compose-next-hidden` parsing and report fields.
2. Extend the rank route builder to optionally emit `route_slot`.
3. Add CUDA kernels:
   - route-slot hidden-shard reduction from `d_down`;
   - deterministic residual shard initialization;
   - float next-hidden composition and finite/checksum reduction.
4. Add a helper that loads a named F8 dense TP tensor from production pack
   bytes and leaves the per-rank output resident on GPU.
5. Use that helper for:
   - `blk.2.attn_output_b.weight`;
   - `blk.2.ffn_down_shexp.weight`.
6. Add explicit peer return from every source contribution shard to every
   destination rank.
7. Compose next-hidden shards on all eight GPUs.
8. Validate:
   - all next-hidden values are finite for the checked sample;
   - repeat composition is deterministic;
   - all per-rank checksums are non-zero;
   - peer return byte count matches expectation;
   - existing dense/KV/EP scaffold still passes.
9. Build and run on the V100 pod.
10. Copy evidence to
    `logs/from-cluster/sprint239-tp-ep-next-hidden-composition/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | TP/EP next-hidden composition mode |
| `docs/sprints/SPRINT-239.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint239-tp-ep-next-hidden-composition/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Implementation stays in the separate TP/EP codepath.
- [ ] No PP scheduler files are modified.
- [ ] `--compose-next-hidden` builds on the V100 pod.
- [ ] Route-to-slot mapping matches the existing EP route counts.
- [ ] TurboMind EP down output is reduced into hidden shards.
- [ ] Expert shard contributions are peer-copied across all eight GPUs.
- [ ] Next-hidden shards include dense attention output, shared FFN output,
      EP contribution, and residual input.
- [ ] The run reports contribution bytes, peer return bytes, composition time,
      finite/checksum status, and final `PASS`.
- [ ] Existing combined dense coverage and KV scaffold checks still pass in the
      same run.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint239-tp-ep-next-hidden-composition/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- The first peer-return path uses float contribution buffers for observability,
  so it is intentionally heavier than the likely production half path.
- This sprint proves TP/EP data movement and composition, not final DS4
  attention semantics.
- The current dense kernels are correctness-oriented scalar CUDA paths, not the
  final HMMA/CUTLASS path.
- If peer copies expose topology or synchronization issues, stop on the
  material uncertainty and record the failure rather than adding PP fallback
  work.

## Decision

Pending.
