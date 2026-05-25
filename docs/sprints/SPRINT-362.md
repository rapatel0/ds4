---
sprint: 362
title: TP/EP Profile Harness Launcher Defaults
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 362 - TP/EP Profile Harness Launcher Defaults

## Overview

Sprint 359 promoted fused compressed pool+norm as a launcher default, and
Sprint 360/361 validated that default through launcher-started HTTP paths. The
permanent profile harness still forces
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=0` unless the
`--fused-compressed-pool-norm` flag is passed, which makes default launcher
tests diverge from production behavior.

This sprint updates `tools/ds4-v100-tp-ep-profile.py` so HTTP launcher runs
respect the launcher default by default, while still allowing explicit A/B
control with a disable flag.

No PP/layer-split work. No MTP. This is a harness correctness cleanup so
future TP/EP serving measurements use the same defaults operators get.

## Implementation

1. Add `--disable-fused-compressed-pool-norm`.
2. Reject using `--fused-compressed-pool-norm` and
   `--disable-fused-compressed-pool-norm` together.
3. In HTTP launcher mode:
   - `--fused-compressed-pool-norm` sets the env var to `1`.
   - `--disable-fused-compressed-pool-norm` sets the env var to `0`.
   - neither flag omits the env var so the launcher default applies.
4. Keep direct-token-major behavior unchanged: direct fused pool-norm is still
   opt-in via the existing `--fused-compressed-pool-norm` flag.

## Verification

- Local Python syntax check passes.
- Local diff whitespace check passes.
- V100 profile harness syntax check passes.
- V100 HTTP `--print-command`/command artifact with no pool flag includes the
  pool-norm gate through the launcher default.
- V100 HTTP command artifact with `--disable-fused-compressed-pool-norm` omits
  the pool-norm gate.

## Definition of Done

- [x] Harness tri-state behavior is implemented.
- [x] Local checks pass.
- [x] V100 command/default proof passes.
- [x] Results are summarized in this sprint doc.
- [x] `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` are updated.
- [x] A temp status report is written.
- [x] Artifacts are committed.

## Outcome

Updated `tools/ds4-v100-tp-ep-profile.py`:

- `--fused-compressed-pool-norm` still forces
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=1`.
- New `--disable-fused-compressed-pool-norm` forces
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=0`.
- Omitting both flags now omits the env var, so the launcher default applies.
- The two flags are mutually exclusive.

V100 selected-token proof, `1` request / `1` token at `32` slots / `256K` /
`position=262143`:

| Harness mode | HTTP 200 | Command has pool gate | Fused pool layers |
|---|---:|---:|---:|
| default | `1/1` | `true` | `40` |
| `--disable-fused-compressed-pool-norm` | `1/1` | `false` | `0` |

The default proof initially used `max_requests=2`, which let the server exit
before the harness fetched `/status`. The validation run uses `max_requests=40`
so post-request status and metrics remain available.

## Decision

The profile harness now matches production launcher defaults for HTTP serving
runs. Future TP/EP HTTP measurements can use the harness default for the
current production behavior and `--disable-fused-compressed-pool-norm` for
explicit control A/B runs.

Artifacts:

```text
logs/from-cluster/sprint362-profile-launcher-defaults/
```
