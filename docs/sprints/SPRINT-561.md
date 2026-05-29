# Sprint 561 - C1 Emitted Topology Graph-Stability

Date: 2026-05-29

## Goal

Advance C1 full-capture reuse by making compressed-KV emitted/non-emitted work
selection graph-stable without adding a permanent flag matrix or promoting
compressed KV serving defaults.

## Context

Sprint 560 fixed the replay-time host metadata half of emitted compressed rows:
a no-suffix full-capture replay now mirrors eager compressed attention/indexer
row counters, row positions, and loaded flags after a successful cache-hit
replay. The remaining steering item is topology: the CUDA graph still captures
different node sets depending on whether a layer position is emitted.

Current host-selected topology in `engine/compressed_kv_step.cu`:

- attention compressed-row emit kernels are under `if (emitted)`
- compressed attention typed-KV store/load is under `typed_compressed && emitted`
- ratio-4 indexer compressed-row emit kernels are under `if (emitted)`
- indexer typed-KV store/load and top-k scoring/broadcast are under
  `typed_indexer && emitted` / `if (emitted && d_indexer_topk)`
- ratio-4 compressed-state shift is under `if (emitted && ratio == 4)`

Sprints 551-554 made physical row and bounded-row selection device-position
dynamic. Sprint 560 made replay host metadata match eager for emitted rows. This
sprint can now focus on making graph-mode enqueue topology independent of the
host emitted boolean while preserving eager output and non-emitted semantics.

## Constraints

- No permanent new CLI/env flag. One-off smokes or temporary validation binaries
  are allowed and must not be committed as accumulated test surface.
- Do not touch MTP.
- Keep promoted suffix replay and served compressed-KV-off defaults unchanged.
- Keep full capture diagnostic-only and position-keyed until cross-position
  cache-key relaxation is separately validated.
- Use previous promoted artifacts as controls unless a real invalidator exists.

## Plan

1. Add a small graph-mode topology helper in `engine/compressed_kv_step.cu` so
   graph capture can enqueue the emitted-row topology consistently while eager
   keeps the existing compact path.
2. Preserve non-emitted semantics explicitly. In graph mode, any extra emitted
   kernels/copies on a non-emitted position must either be device-masked or
   write only ignored scratch/state; indexer top-k visible to attention must not
   change on non-emitted positions.
3. Keep host metadata updates tied to the real host `emitted` value. Do not
   increment row counters or mark rows loaded for non-emitted positions.
4. Validate with a one-off compressed-KV binary at two same-shape positions:
   - emitted ratio-4 position `262083`
   - adjacent non-emitted ratio-4 position `262084`
5. Compare compressed-KV eager vs no-suffix full-capture replay for selected
   tokens and decode checksums. The replay leg must show `43/43` cache hits,
   zero invalidations, zero peer/SYS transport, and zero NCCL graph SYS edges.

## Definition of Done

- [x] Remote build passes in the workload container.
- [x] Served-default compressed-KV-off no-regression remains clean enough to show
  promoted suffix/default behavior is not affected.
- [x] One-off compressed-KV emitted-position eager and replay probes match selected
  tokens and decode checksums.
- [x] One-off compressed-KV non-emitted-position eager and replay probes match
  selected tokens and decode checksums, proving extra graph-stable topology does
  not perturb non-emitted semantics.
- [x] Full-capture replay remains position-keyed; this sprint does not relax the
  cache key.
- [x] `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` are updated with the
  decision and next ordered item.
- [x] No temporary smoke source, temporary binary target, or new production flag is
  committed.

## Implementation

- Added a graph-mode topology policy in `engine/compressed_kv_step.cu`:
  eager keeps the compact host-selected `emitted` branches, while graph mode
  enqueues the emitted-row compressed-KV topology for any compressed ratio.
- Added device-side emitted-position guards for compressed attention/indexer
  emit, score, top-k copy, and ratio-4 shift kernels. The guards read
  `RankState::d_decode_position`, so a captured graph can replay at emitted
  and non-emitted positions without changing its node set.
- Kept host-visible row counters, row positions, and loaded-row metadata gated
  by the real host `emitted` boolean. Non-emitted graph-mode extra work is
  either device-masked or writes state that is not made visible by host row
  metadata.
- Added no permanent CLI/env flag and no committed temporary smoke source.

## Validation

Remote build:

- `/workspace/s561-emitted-topology`
- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Served-default compressed-KV-off sanity:

- Artifact:
  `/workspace/s561-emitted-topology-artifacts-user/none-s561-served-default-standard-seq2-serverargs-h396a9fa7`
- `http_200=2`
- `compressed_kv_layers=0`
- `scaffold_decode_cudagraph_replay_succeeded=43`
- `scaffold_decode_cudagraph_persistent_cache_hits=43`
- `scaffold_decode_cudagraph_persistent_invalidations=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`

One-off compressed-KV emitted position, `position=262083`:

- Eager artifact:
  `/workspace/s561-emitted-topology-artifacts-user/none-s561-compressed-emitted-eager-seq2`
- No-suffix full-capture replay artifact:
  `/workspace/s561-emitted-topology-artifacts-user/none-s561-compressed-emitted-fullgraph-seq2-serverargs-h396a9fa7`
- Both runs exercised `compressed_kv_layers=86` and
  `compressed_kv_emitted_layers=42`.
- Selected tokens matched exactly: `58204`, `109597`.
- Decode checksums matched exactly: `7265791446`, `79399742586`.
- Replay leg: `43/43` cache hits, zero invalidations, zero peer/SYS, zero NCCL
  graph SYS edges.

One-off compressed-KV adjacent non-emitted position, `position=262084`:

- Eager artifact:
  `/workspace/s561-emitted-topology-artifacts-user/none-s561-compressed-nonemitted-eager-seq2`
- No-suffix full-capture replay artifact:
  `/workspace/s561-emitted-topology-artifacts-user/none-s561-compressed-nonemitted-fullgraph-seq2-serverargs-h396a9fa7`
- Both runs exercised `compressed_kv_layers=86` and
  `compressed_kv_emitted_layers=0`.
- Selected tokens matched exactly: `109939`, `107875`.
- Decode checksums matched exactly: `561376577`, `2841198172`.
- Replay leg: `43/43` cache hits, zero invalidations, zero peer/SYS, zero NCCL
  graph SYS edges.

Claude bug-find review:

- Ran `$claude` bug-find against the Sprint 561 diff.
- Main risk identified: graph-mode typed-KV store/load now runs under the
  graph-stable topology at non-emitted positions. The non-emitted parity gate
  above directly covered that risk and matched eager selected tokens/checksums.

## Decision

Promote as C1 full-capture readiness/correctness repair only. No-suffix full
capture remains diagnostic-only and position-keyed. The next ordered work is
cross-position full-capture cache-key relaxation as a diagnostic parity sprint,
not serving promotion.
