# Sprint 045: Production MTP Speculative Serving

## Status

Complete.

## Overview

Sprint 045 closes the current `mtp_speculative_serving` readiness blocker by
exposing the already-gated native one-token MTP verify path through the resident
HTTP appliance. The first served mode should be conservative: base generation
remains the verifier, MTP drafts are accepted only when they exactly match the
target top-1 token, and status/metrics must make the mode obvious.

This sprint should not claim multi-slot throughput or full speculative speedup.
The goal is production-serving correctness and observability for MTP inside the
same resident process operators start.

## Use Cases

- Start the appliance with MTP serving enabled from the normal launcher path.
- Probe `/v100/status` and see `mtp_enabled=true`,
  `speculative_serving=true`, and the MTP mode/top-k.
- Send a bounded loopback generation request and receive base tokens plus MTP
  draft/verify diagnostics.
- Preserve the base one-slot path and fallback to `mtp_enabled=false` by
  default.
- Keep full gate readiness honest: after this sprint passes, the next blocker
  should be multi-slot or aggregate throughput, not MTP serving.

## Architecture

- Keep the base one-slot replay runtime as the request owner.
- Add narrow replay accessors instead of exposing scheduler internals broadly:
  - read committed token embedding as F32 from stage 0;
  - read gpu7 post-commit HC;
  - expose the mapped base model pointer/size for MTP output-head upload.
- Add a resident MTP service object in `tools/ds4-v100-replay`:
  - opens the MTP sidecar on gpu7;
  - opens `ds4_v100_mtp_forward` once at server startup;
  - keeps MTP top-k and counters;
  - runs one draft/verify after base generation has produced at least two
    tokens.
- Verification policy for this sprint:
  - committed token is the first generated base token;
  - target token is the second generated base token;
  - MTP draft input is committed-token embedding plus post-commit target HC;
  - accept only if `mtp_top1 == target_top1`;
  - do not mutate target scheduler state for accepted tokens yet.
- JSON response includes an `mtp` object when MTP serving is enabled.
- Metrics include MTP enabled/drafts/accepted/rejected counters.
- Launcher/env/gate gain an explicit MTP serving mode, defaulting to off.

## Parallel Work

Parallel sidecar agents should inspect:

- serving/request-loop status, metrics, and reset semantics;
- extraction path from `tools/ds4-v100-mtp-verify-smoke.c` into a reusable
  resident service helper.

## Implementation

1. Add replay accessors in `ds4_v100_replay.h/.c`.
2. Extend `tools/ds4-v100-replay` with `--mtp-model`,
   `--mtp-serving off|verify`, `--mtp-top-k`, and MTP response/metrics output.
3. Link the replay tool with the existing MTP sidecar and forward-common
   implementation.
4. Update `tools/ds4-v100-run-appliance.sh` and env examples to pass MTP
   serving mode explicitly.
5. Add `tools/ds4-v100-mtp-serving-smoke.sh` for loopback status/metrics/
   generation validation.
6. Wire the smoke into `tools/ds4-v100-gate.sh` as
   `mtp_speculative_serving`.
7. Update operations docs, sprint report, follow-ups, and vision.

## Files Summary

- `ds4_v100_replay.h`
- `ds4_v100_replay.c`
- `tools/ds4-v100-replay.c`
- `tools/ds4-v100-mtp-serving-smoke.sh`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-gate.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `docs/sprints/SPRINT-045-REPORT.md`
- `docs/sprints/SPRINT-045-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local object compile passes for changed C files.
- Shell syntax checks pass.
- `tools/ds4-v100-replay --help` documents MTP serving flags after cluster
  build.
- Base serving with MTP off still passes existing appliance and production
  deployment smokes.
- On the V100 cluster, MTP serving mode starts with the real MTP sidecar,
  reports `mtp_enabled=true`, exposes MTP metrics, and returns first token bytes
  `3136`.
- Served response includes MTP draft/verify diagnostics with exact top-1 accept
  on the short fixture.
- Full V100 gate includes `mtp_speculative_serving PASS`, has no failures, and
  no longer reports `missing=mtp_speculative_serving`.
- Sprint report records commands, outputs, artifacts, timings, and remaining
  readiness blockers.

## Risks

- The first served mode verifies against the base target path and may not reduce
  latency yet. That is acceptable for this sprint; correctness and transaction
  visibility come first.
- MTP sidecar plus output-head upload adds gpu7 memory pressure. Require the
  same reserve checks used by the MTP verify smoke.
- `ds4_v100_replay_generate` leaves state after the first generated token when
  two tokens are requested. The MTP serving path must rely only on that
  documented state or add an explicit replay API for post-commit state.
- Accepting tokens without target mutation would be misleading. This sprint
  reports acceptance diagnostics but still returns the base-verified token.

## Security

No new external exposure. MTP serving remains loopback-only and bounded by the
existing one-slot, sequential request model.

## Dependencies

- Sprint044 optimized replay open/upload and gate.
- Sprint042 MTP verify/rollback correctness.
- Real base model, MTP sidecar, pack index, and 8x V100 cluster access.

## Outcome

`SHIP`. Sprint 045 exposes the existing gated one-token MTP verify path through
the resident HTTP appliance as `--mtp-serving verify` / `DS4_V100_MTP_SERVING=verify`.
The base path remains the default rollback mode. The full V100 gate now passes
`mtp_speculative_serving` and advances readiness to
`missing=aggregate_slot_context_envelope`.

## Resolved Questions

- The first served MTP mode skips diagnostics when fewer than two generated
  tokens are available. The MTP serving smoke requires `tokens >= 2`.
- Accepted draft counters count exact diagnostic matches against the base target
  token. A later sprint must distinguish this from true committed speculative
  speedup.
