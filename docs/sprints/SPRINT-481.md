# Sprint 481 - Code Cleanup After Pattern-A Promotion

## Overview

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` has become a matrix of historical
feature gates. This sprint snapshots the current state, then removes dead
branches from the promoted TP/EP serving path while reviving the verified A6
rank-major attention norm path.

This is a destructive cleanup sprint. The first executable step is a pushed
snapshot commit and tag.

## Use Cases

- Engineers can reason about the promoted TP/EP serving path without traversing
  rejected, closed-audit, or default-off historical branches.
- Promotion and rejection decisions immediately collapse old flags instead of
  accumulating branch debt.
- A6 PATH 4 uses the existing rank-major current buffer for attention projection
  input norm when available, without changing math.

## Architecture

Use the six buckets from `docs/sprints/archive/TEMP_CODE_CLEANUP_PROMPT.md`:

- Promoted: delete the flag and dead else branch.
- Rejected terminal: delete the flag and dead if branch.
- Dormant revive: enable, validate, then delete broken siblings.
- Diagnostic/audit: delete unless the audit is open; otherwise isolate behind
  diagnostics.
- Runtime knob: keep and document.
- Experimental alive: keep with owner/sunset comment.

## Implementation

### Phase 0 - Snapshot

- Stage all current tracked and untracked sprint work explicitly.
- Commit: `Pre-cleanup snapshot: state before TEMP_CODE_CLEANUP_PROMPT`.
- Push to `origin/claude-takeover`.
- Tag the commit `pre-cleanup-snapshot` and push the tag.
- Record the SHA in `docs/sprints/archive/status-reports/TEMP_STATUS_REPORT_481.md`.

### Phase 1 - A6 PATH 4 Revive

- In `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, replace the hardcoded
  attention-projection `rank_major_input=false` with the promoted rank-major
  current-buffer condition, guarded by `r.d_current_full_rank_major`.
- Audit that `r.d_current_full_rank_major` remains live until the norm kernel
  reads it.
- Validate strict reference-shape parity.

### Phase 2 - A6 Sibling Deletion

- Delete the no-op broadcast sibling and broken direct/rank-local sibling once
  PATH 4 is the reachable fast path.
- Remove `--true-ds4-attention-projection-direct-input-fill-gate`,
  `--true-ds4-attention-projection-rank-local-input-gate`, and the associated
  launcher/profile/harness wiring if no longer meaningful.
- Validate strict reference-shape parity.

### Phase 3 - Transport Dead-Branch Deletion

- Remove retired `ds4_peer_copy_async` fallback branches in the promoted
  serving path.
- Remove or collapse `enqueue_graph_f32_copy_from_device0`,
  `enqueue_graph_f32_copy_between_devices`, and
  `--decode-cudagraph-peer-copy-gate`.
- Validate strict reference-shape parity and `peer_copy_sys_bytes=0`.

### Phase 4 - Parity/Diagnostic Cleanup

- Remove closed `*-parity-gate` paths, parser entries, scaffold fields, and
  profile/harness summaries.
- Keep peer accounting counters grouped as diagnostics.
- Keep real runtime knobs such as `--nccl-reduce-scatter-compose-gate`.
- Validate strict reference-shape parity.

### Phase 5 - Surviving Flag Inventory

- Count flags and file lines before/after.
- Document every surviving flag by bucket with one-line justification.
- Add going-forward flag discipline to `docs/sprints/VISION.md`.
- Write `docs/sprints/archive/status-reports/TEMP_STATUS_REPORT_481.md`.

## Files Summary

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`: primary cleanup target.
- `tools/ds4-v100-run-appliance.sh`: remove deleted flag env/parser/CLI wiring.
- `deploy/v100/ds4-v100-appliance.env.example`: remove deleted env knobs and
  document retained runtime knobs.
- `tools/ds4-v100-tp-ep-profile.py`: remove deleted profile options and summary
  fields.
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`: remove deleted A/B options and
  summary fields.
- `docs/sprints/VISION.md`: record cleanup discipline and sprint result.
- `docs/sprints/archive/status-reports/TEMP_STATUS_REPORT_481.md`: sprint execution report.

## Definition Of Done

- Pre-cleanup snapshot commit pushed to `origin/claude-takeover`.
- `pre-cleanup-snapshot` tag pushed.
- A6 PATH 4 is enabled or explicitly backed out with evidence if parity fails.
- Deleted buckets no longer appear in the promoted hot-path control flow.
- Strict reference-shape parity passes after cleanup commits:
  `32` slots / `256K` / `256` requests / `64` tokens.
- `peer_copy_sys_bytes=0` confirmed.
- Local syntax/build checks pass for touched code.
- End-of-sprint report records snapshot SHA, flag count before/after, line
  count before/after, validation artifacts, and surviving flag inventory.

## Risks

- A branch classified as dead may still be reachable under the promoted
  serving configuration. The strict parity gate catches this and requires
  backout/reclassification.
- A6 PATH 4 may not be bit-exact despite the prompt's expectation; if so,
  back out the revive and leave the cleanup of its siblings scoped to dead code
  only.
- Full reference-shape validation is slow. Keep commits coherent so failures
  can be attributed.

## Security

No external service behavior changes are intended. The main operational risk is
shipping a changed serving path without strict parity; do not accept tolerance
for this sprint.

## Dependencies

- V100 pod `ds4-tp-bench` for build and strict reference-shape parity.
- Existing appliance pack and contract paths used by recent Sprint 480 sanity.
- Git remote `origin/claude-takeover` available for the required snapshot push.

## Open Questions

- How many cleanup buckets can be safely grouped per commit given the cost of
  the strict gate?
- Which remaining diagnostics, if any, have an active owner after Sprint 480?

## Execution Notes

- Snapshot commit `e65614cb` was pushed and tagged `pre-cleanup-snapshot` before
  destructive cleanup began.
- A6 PATH 4 revive was attempted but not promoted. The control baseline reached
  `256/256` HTTP 200 with the in-pod harness, but reported
  `peer_copy_sys_bytes=8`; the A6 candidate returned `0/256` HTTP 200 after
  readiness. The A6 diff was backed out before continuing cleanup.
- Root cleanup landed in two commits:
  - `14c773f2`: archived old numbered status reports.
  - `d01917a6`: archived superseded root topic docs.
- `tools/ds4-source-oracle-vector.{c,o}` was retained after audit because it is
  still referenced by `Makefile` and `tools/ds4-v100-gate.sh`.
- Code cleanup landed in `df9250e8`, removing the retired
  `--decode-cudagraph-peer-copy-gate` option/status plumbing. V100 build passed
  afterward.
