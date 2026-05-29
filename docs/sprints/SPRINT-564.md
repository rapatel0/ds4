# Sprint 564 - C1 No-Suffix Cache-Miss Device-State Repair

Date: 2026-05-29

## Goal

Localize and repair the next no-suffix full-capture cross-position replay
hazard after Sprint 563 showed the first logged divergence at layer 1
`hc_current`.

## Context

Sprint 563 localized the failed cross-position retry to layer-1 HC-current
input state: layer 0 still matched, but occurrence 1 first diverged at layer 1
`hc_current`.

The first hypothesis was that the cache-miss path itself advanced live device
state twice:

- `run_decode_loop()` sees no-suffix replay-probe plus no persistent cache hit.
- It first calls `run_eager_decode_steps()`.
- It then calls `attempt_capture_probe(true)`.
- The capture path enqueues `run_one_step()` to build the CUDA graph, which
  executes kernels on the same live device buffers.
- Because no replay is launched on full-capture cache miss, only host metadata
  is restored. The device state remains advanced by the capture execution.

Validation rejected that hypothesis. A local candidate that skipped eager and
treated stream capture as the served result returned token `0` instead of
eager token `24426`, proving that stream capture does not materialize the live
tensors. Cache miss must continue to serve eager and only instantiate the
captured graph for a later replay.

## Constraints

- No permanent new CLI/env flag.
- Do not touch MTP.
- Keep promoted suffix replay/default behavior unchanged.
- Keep no-suffix full capture diagnostic-only and position-keyed unless the
  repaired path is later validated for cache-key relaxation.
- Do not reintroduce immediate replay-after-capture on live buffers.

## Plan

1. Reject capture-as-result if validation shows stream capture does not
   execute the recorded kernels.
2. Add graph-cache metadata for full-capture final-HC buffer identity so a
   cached graph cannot silently replay against a different host-visible
   `d_final_hc_shard` pointer.
3. Build on the V100 node.
4. Validate:
   - same-session no-suffix position-keyed diagnostic still matches eager,
   - a remote-only relaxed-position retry shows whether the final-HC pointer
     key fixes the Sprint 563 divergence,
   - promoted suffix/default sanity remains clean.

## Result

Implemented a defensive full-capture cache key extension:

- `TpCudaGraphLayerExec::final_hc_shard_key` records the host-visible
  `RankState::d_final_hc_shard` pointer at capture time.
- The persistent invalidation path now rejects no-suffix full-capture replay
  if the live final-HC pointer key differs from the captured one.
- The caller-side full-capture cache-hit precheck uses the same key so a key
  miss still runs eager before capture. This preserves the cache-miss contract:
  stream capture records the graph but does not serve the response.

Rejected the original cache-miss hypothesis:

- Candidate skipped eager and treated capture as the served result.
- Validation failed immediately: graph request 1 returned selected token `0`
  / checksum `5074584678` while eager returned token `24426` / checksum
  `128829740021`.
- The candidate was reverted before the committed fix.

Validation:

- Remote build passed in `/workspace/s564-cache-miss-state`.
- Position-keyed no-suffix same-session diagnostic matched eager for three
  selected-token requests:
  - request 1: `24426` / `128829740021`
  - request 2: `2039` / `106648190597`
  - request 3: `117465` / `17092309830`
- Remote-only relaxed-position retry with the final-HC pointer key still
  failed on request 3:
  - eager: `117465` / `17092309830`
  - relaxed replay: `128818` / `81184816026`
- Promoted suffix/default sanity passed via
  `/workspace/s564-cache-miss-state-profile-artifacts/none-s564-served-default-suffix-sanity`:
  `http_200=2`, `43/43` suffix replay hits on the second request, zero
  persistent invalidations, zero peer-copy/SYS, zero NCCL graph SYS edges,
  `compressed_kv_layers=0`, and `graph_audit_blocker=none`.

## Decision

Promote the final-HC pointer key as a defensive correctness guard. Do not
relax the full-capture position key. The request-3 failure with the remote-only
relaxed build shows the remaining blocker is still captured position-dependent
state, most likely scalar `opt.position` / derived row arguments baked into
captured kernels outside the `d_decode_position` device path.

## Definition of Done

- Remote build passes in the workload container.
- The no-suffix cache-miss path continues to serve eager; capture is not used
  as a live response.
- Same-session no-suffix position-keyed diagnostic responses match eager
  selected tokens/checksums for the three-request shape that Sprint 563 used.
- Promoted suffix/default sanity remains clean.
- `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` are updated with the
  result and next ordered item.
- No temporary source, temporary binary target, or new production flag is
  committed.
