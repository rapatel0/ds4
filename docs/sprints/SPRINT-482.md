# Sprint 482 - A6 PATH 4 Failure Capture

## Overview

Sprint 481 attempted to revive A6 PATH 4, the rank-major attention projection
input norm path, but backed it out after the candidate returned `0/256` HTTP
200 responses. This sprint narrows the next A6 step to observability and a
diagnostic-only switch. It must not promote A6 or rerun broad reference A/Bs
until the failure mode is captured.

## Use Cases

- Engineers can reproduce the A6 PATH 4 candidate without editing CUDA source.
- A failed A6 readiness or request run leaves a compact summary with server
  stdout/stderr tails, lifecycle state, command, env, and return code.
- The promoted TP/EP appliance defaults remain unchanged while A6 is debugged.

## Architecture

Keep the existing rejected `--true-ds4-attention-projection-rank-local-input`
path untouched. Add a separate diagnostic switch for PATH 4 that:

- requires the HC-current NCCL allgather rank-major buffer,
- routes attention projection input fill through
  `fill_two_hidden_inputs_half_from_rank_major_norm_kernel`,
- bypasses the canonical GPU0 normed-current broadcast for this diagnostic
  branch only,
- leaves the old rank-local sibling available for historical comparison.

The profile harness should write `failure-summary.json` and
`failure-summary.md` when the server exits before readiness or fails before a
normal `summary.json` can be produced.

## Implementation

1. Add `--true-ds4-attention-projection-rank-major-input-gate` to the TP/EP
   serving binary as a diagnostic-only flag.
2. Add launcher/profile/env wiring:
   `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT=0` and
   `tools/ds4-v100-tp-ep-profile.py --attention-projection-rank-major-input`.
3. Make the attention projection dispatch use PATH 4 only when the new flag and
   `tp_hc_current_input_nccl_allgather_gate` are both active and
   `r.d_current_full_rank_major` exists.
4. Add profile early-failure summary capture before rerunning A6.

## Files Summary

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`: diagnostic flag and PATH 4
  dispatch.
- `tools/ds4-v100-run-appliance.sh`: env validation and command wiring.
- `tools/ds4-v100-tp-ep-profile.py`: profile flag and failure summaries.
- `deploy/v100/ds4-v100-appliance.env.example`: documented diagnostic env.
- `docs/sprints/SPRINT-482.md`: sprint record.

## Definition Of Done

- Local syntax checks pass for touched Python and shell files.
- Local CUDA stub or remote CUDA build command is run when available.
- `--print-command` with the new env shows the PATH 4 diagnostic flag and the
  required HC-current NCCL allgather flag.
- No appliance default changes: the new env defaults to `0`.
- A6 is still classified as rejected/pending-debug until a later V100 run
  captures and fixes the server failure.

## Risks

- PATH 4 may still fail before readiness. That is expected for this sprint; the
  new artifact summary is the deliverable.
- The old rank-local sibling and the new rank-major diagnostic can be confused.
  Keep names distinct and keep the old default off.
- A broad A/B without server-side failure capture would repeat Sprint 481's
  ambiguity. Do not run one as part of this sprint.

## Security

No external exposure changes. The sprint only adds local diagnostic flags and
artifact capture.

## Dependencies

- Existing Sprint 481 cleanup baseline.
- V100 pod build for final CUDA validation if available.
- Existing `tools/ds4-v100-run-tp-ep-appliance.sh` launcher and profile harness.

## Open Questions

- Did the Sprint 481 PATH 4 candidate fail during startup allocation, first
  decode, or HTTP request handling?
- Does PATH 4 need an additional stream/event dependency after HC-current
  allgather, or is the failure a missing buffer/weight allocation?

## Execution Record

- Initial V100 diagnostic artifact:
  `/workspace/s482-a6-path4-diagnostic/none-peeracct-rejectsys-attn-proj-rank-major-nccl-no-sys-parallel-expert-load`.
  It exited before readiness with `server exited rc=2`; the captured command
  still included `--true-ds4-semantic-skip-stats-gate` even though the A6
  diagnostic did not request attention-output or post-attention semantic
  serving.
- Cleanup fix: profile env export and launcher command wiring now only emit
  `--true-ds4-semantic-skip-stats-gate` when attention-output,
  attention-output NCCL allgather, or post-attention FFN input is active.
  This keeps the promoted semantic skip-stats path available while preventing
  non-semantic diagnostics from tripping binary validation.
- Updated files were recopied to the V100 node via
  `rsync` over `ssh ubuntu@192.168.102.5` after the node path was confirmed as
  `/workspace -> /localpool/ds4/workspace`.
- Rerun artifact:
  `/workspace/s482-a6-path4-diagnostic-rerun/none-peeracct-rejectsys-attn-proj-rank-major-nccl-no-sys-parallel-expert-load`.
  The command contained `--true-ds4-attention-projection-rank-major-input-gate`
  and `--tp-hc-current-input-nccl-allgather-gate`; the semantic skip-stats
  command flag count was `0`.
- Rerun result: server ready at `106.177856s`, selected-token requests
  completed `4/4` HTTP 200 in `2.7812509536743164s`, `resident_ready=true`,
  `warmed_ready=true`, `rejected_requests=0`, `total_generated_tokens=4`,
  `peer_copy_ops=0`, `peer_copy_sys_bytes=0`, and `vram_failures=0`.
- Sprint conclusion: the observed Sprint 482 blocker was stale cleanup-era
  launcher/profile wiring, not a confirmed PATH 4 startup crash. A6 remains
  default-off and still needs a same-binary tolerance/promotion A/B before any
  promotion decision.
