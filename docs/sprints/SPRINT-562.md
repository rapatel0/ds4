# Sprint 562 - C1 Full-Capture Cross-Position Cache-Key Relaxation

Date: 2026-05-29

## Goal

Retry no-suffix full-capture cross-position cache reuse now that device
position consumers, replay host metadata, and compressed-KV emitted topology
are graph-stable. This is a diagnostic parity sprint only, not no-suffix
serving promotion.

## Context

Sprint 555 proved that dropping the full-capture position key too early was
unsafe. Since then:

- Sprint 558 repaired no-suffix full-capture validation so cache miss serves
  eager and cache hit replays only from fresh request state.
- Sprint 559 mirrored the final-HC host pointer swap after full-capture replay.
- Sprint 560 mirrored emitted compressed-row host metadata after replay.
- Sprint 561 made graph-mode compressed-KV emitted/non-emitted topology stable
  and validated emitted plus adjacent non-emitted positions against eager.

The remaining steering item is to retry cross-position cache-key relaxation.
The key implementation detail is that the no-suffix replay-probe precheck must
agree with the persistent cache invalidation policy: if full capture is allowed
to reuse across positions, the precheck must not run eager before launching the
cached graph.

## Constraints

- No permanent new CLI/env flag.
- Do not touch MTP.
- Keep promoted suffix replay and served defaults unchanged.
- Keep this diagnostic-only. Passing parity does not promote no-suffix full
  capture as a serving default.
- Preserve the previous-promoted run as the control unless a real invalidator
  exists.

## Plan

1. Refactor the CUDA graph persistent-position-key predicate in
   `engine/decode_loop.cu` so the same policy is used by:
   - persistent cache invalidation
   - no-suffix replay-probe cache-hit detection
2. Relax the position key for no-suffix full capture only. Keep existing
   position-key behavior for unsafe suffix shapes; retain the existing
   promoted stable compose-suffix exception.
3. Validate same-shape cross-position reuse through HTTP with repeated
   same-session selected-token requests so the second request starts at the
   next cache position and reuses the cached full graph.
4. Compare eager and full-capture replay selected tokens/checksums. The replay
   leg must show `43/43` cache hits, zero invalidations, zero peer/SYS, and
   zero NCCL graph SYS edges.
5. If cross-position parity fails, remove the candidate and record the
   concrete blocker in this sprint and steering.

## Definition of Done

- [x] Remote build passes in the workload container.
- [x] No-suffix full-capture cross-position replay was tested without position
  invalidation.
- [ ] Eager versus full-capture replay selected tokens and decode checksums match
  for the cross-position diagnostic shape.
- [x] Served promoted suffix/default behavior is not changed.
- [x] `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` record either promotion
  as diagnostic C1 readiness or rejection with the concrete blocker.
- [x] No temporary source, temporary binary target, or new production flag is
  committed.

## Implementation Tested

The candidate removed the no-suffix full-capture position key from both:

- persistent graph invalidation
- no-suffix replay-probe cache-hit detection

No new flag was added. The promoted suffix exception for stable
`compose_eager_final_hc` geometry was left unchanged.

## Validation

Remote build:

- `/workspace/s562-cross-position`
- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Two-request same-session probe:

- Artifacts:
  `/workspace/s562-cross-position-artifacts-user/s562-crosspos-eager-same-session`
  and
  `/workspace/s562-cross-position-artifacts-user/s562-crosspos-fullgraph-same-session`
- Eager and no-suffix full-capture replay matched selected tokens/checksums
  for the first two same-session requests:
  - tokens: `24426`, `2039`
  - checksums: `128829740021`, `106648190597`
- Replay logs showed `43` cache hits on request two, with
  `position=262084` reusing `cached_position=262083`, and zero position
  invalidations.

Six-request same-session probe:

- Artifacts:
  `/workspace/s562-cross-position-artifacts-user-6/s562-crosspos6-eager-same-session`
  and
  `/workspace/s562-cross-position-artifacts-user-6/s562-crosspos6-fullgraph-same-session`
- This stronger gate intentionally crossed multiple adjacent positions from
  the original captured position.
- It failed on request three:
  - eager token/checksum: `117465` / `17092309830`
  - replay token/checksum: `2039` / `110810249310`
- Subsequent requests also diverged, so the candidate was removed.

Promoted suffix/default sanity:

- Artifact:
  `/workspace/s562-cross-position-profile-artifacts/none-s562-served-default-suffix-sanity`
- `http_200=2`
- `compressed_kv_layers=0`
- `scaffold_decode_cudagraph_replay_succeeded=43`
- `scaffold_decode_cudagraph_persistent_cache_hits=43`
- `scaffold_decode_cudagraph_persistent_invalidations=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`

Claude bug-find review:

- Flagged a likely next blocker: full-capture still bakes some position-derived
  row/source choices into captured graph arguments even though the top-level
  `d_decode_position` consumers and emitted topology are graph-stable.
- The concrete six-request failure confirms cross-position full-capture reuse
  still has hidden position-captured state before it can be promoted even as a
  diagnostic cache-key relaxation.

## Decision

Rejected. The candidate code was removed. Full capture remains position-keyed.
The next C1 work is not another blind cache-key retry; it must localize and
convert the remaining captured position-derived graph arguments/state that
cause divergence after more than one cross-position replay.
