# Sprint 445: Rank-Major TP/EP Serving Decision

## Objective

Stay TP/EP only and produce the clean rank-major HTTP serving A/B that Sprint
443 could not finish because stale graph-server processes interrupted the run.

Sprint 444 fixed process hygiene and request budgeting. This sprint uses that
cleanup to decide whether the current rank-major candidate is promotable in
serving, or whether the next performance work should move to another TP/EP
lever.

## Implementation Plan

1. Sync the current TP/EP runtime and harness files to the V100 workspace.
2. Rebuild the TP/EP full-layer smoke/server binary on gpu-01.
3. Verify no stale DS4 GPU jobs are running.
4. Run the same-binary HTTP A/B at the Sprint 443 rank-major shape:
   - `8` requests;
   - `8` slots;
   - `256K` context;
   - `2` generated tokens/request;
   - chat endpoint;
   - control: HC-current NCCL, post-attention FFN input, fixed-capacity route
     plan;
   - candidate: control plus rank-local/rank-major attention input,
     rank-major FFN input, and rank-major router logits.
5. Require readiness, response parity, VRAM admission, and a server-decode
   speedup before promotion.

## Validation

- Local Python syntax checks for the profile and A/B harness.
- Remote build of `tools/ds4-v100-tp-ep-full-layer-smoke`.
- V100 HTTP A/B artifact with readiness, response parity, GPU samples, and
  parsed metrics.
- No stale DS4 GPU jobs after the run.

## Execution Notes

The first clean-window attempt failed before readiness because `/tmp/ds4.lock`
was left behind as a root-owned file by an earlier sudo/root profile. The
profile harness now sets `DS4_LOCK_FILE` to the per-case artifact directory so
serving runs are isolated from stale global lock files.

## Out Of Scope

- PP/layer-split work.
- MTP.
- Promoting persistent graph serving.
- Host-synchronized actual-route execution.

## Decision Rule

Promote only if:

- control and candidate both return HTTP 200 for all requests;
- readiness checks pass for both legs;
- response parity matches every request;
- VRAM guard failures are zero;
- server generated decode improves by at least `1.02x`.

If the candidate is correct but flat/slower, keep the rank-major gates
diagnostic-only and move the next sprint to either a real full-shape routed FFN
executor or a safer graph replay design with device-side dynamic state.

## Outcome

Artifact:

- `/localpool/ds4/workspace/logs/s445-rank-major-http-ab-lockfix`

The A/B completed cleanly after the per-case lock fix:

| Leg | HTTP 200 | Ready | First token | Server decode tok/s | Continuation tok/s | Client tok/s | Avg GPU util | Min free VRAM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Control | 8/8 | yes | 72960 | 19.279431 | 18.771453 | 1.508636 | 9.840278% | 4674 MiB |
| Candidate | 8/8 | yes | 81401 | 20.362245 | 20.277332 | 1.522924 | 10.173611% | 4836 MiB |

Speed signal:

- Server generated decode: `1.056x`
- Server continuation decode: `1.080x`
- HC-current input time: `246.465094 -> 221.645599 ms`
- HC-current gather time: `4.457132 -> 4.194509 ms`

Validation failure:

- Response parity: `0/8` matched.
- Control first token: `72960`.
- Candidate first token: `81401`.
- Control checksum: `11698534079367350598`.
- Candidate checksum: `11698534088515400375`.

## Decision

Do not promote the combined rank-major serving candidate. It has a real speed
signal and slightly better VRAM headroom, but it changes generated tokens.

The next sprint should isolate the rank-major gates one at a time:

1. rank-local/rank-major attention input only;
2. rank-major FFN input only;
3. rank-major router logits only.

Only a parity-clean single gate can be considered for promotion or further
composition.
