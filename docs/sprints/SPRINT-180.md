# Sprint 180 - MTP Verify Active Microbatch Serving

Date: 2026-05-22
Status: Completed

## Overview

The previous routed-FFN wrapper and TP/EP overlay sprints narrowed the hot path
but did not produce a promotable throughput gain. While reviewing the next
larger execution boundary, one production serving gap stood out: MTP serving was
hard-gated to `active_microbatch=1`, even for conservative `verify` mode.

Sprint 180 makes MTP verify compatible with same-length active microbatching.
This is not true speculative commit throughput; it is a correctness-safe
serving capability that lets the production multi-slot benchmark exercise the
resident MTP sidecar without disabling base batching.

## Non-Goals

- No promotion of MTP commit to multi-slot mode.
- No claim that MTP verify improves throughput.
- No true speculative verifier that drafts multiple tokens and verifies them in
  one base forward.
- No change to default `DS4_V100_MTP_SERVING=off`.

## Architecture

Before:

```text
MTP off:
  pending same-token-count requests -> batched base generation

MTP verify/commit:
  active_microbatch must be 1
```

After:

```text
MTP verify:
  pending same-token-count, same-prompt-length requests
    -> batched base generation
    -> per-request MTP verify while generation mutex is still held
    -> response includes per-request mtp JSON

MTP commit:
  remains active_microbatch=1
```

The same-prompt-length condition keeps the replay batch slot order stable, so
MTP verify can read each request's final HC from the matching scheduler slot.
Verification stays inside the generation critical section so another request
cannot reset the replay runtime before HC is read.

## Implementation

- Add `ds4_v100_replay_read_output_hc_slot`.
- Route existing one-slot HC reads through slot `0`.
- Add an MTP verify slot helper in `tools/ds4-v100-replay.c`.
- Allow `--mtp-serving verify` with `--active-microbatch > 1`.
- Keep `--mtp-serving commit` restricted to `--active-microbatch 1`.
- Enable batch formation for MTP verify only when token count and prompt length
  match across the pending batch.
- Store MTP verify results on each pending request before marking it complete.
- Update operator docs to distinguish verify batching from commit batching.

## Validation

- V100 build:

```text
make ds4_v100_replay.o tools/ds4-v100-replay CUDA_ARCH=sm_70 -j80
```

- CLI guard checks:
  - `--mtp-serving verify --active-microbatch 2` starts.
  - `--mtp-serving commit --active-microbatch 2` fails closed.
- Served tests on the V100 pod:
  - MTP verify, `2` slots, `256K`, `2` generated tokens.
  - MTP verify, `16` slots, `256K`, `2` generated tokens.
  - record prompt/prefill tok/s, generated tok/s, continuation/decode tok/s,
    token match, tensor-batch counters, and MTP request/draft counters.

## Results

Implemented MTP verify active-microbatch serving. `--mtp-serving verify` now
allows `--active-microbatch > 1`, while `--mtp-serving commit` still fails
closed unless `active_microbatch=1`.

The implementation adds a slot-specific final-HC read:

```text
ds4_v100_replay_read_output_hc_slot(rt, slot, ...)
```

and runs MTP verify inside the generation critical section for batched
requests. The batched verify path is intentionally limited to same-token-count,
same-prompt-length request groups so replay slot order remains stable.

V100 build on `llm/llamacpp-build-8gpu` passed:

```text
make ds4_v100_replay.o tools/ds4-v100-replay CUDA_ARCH=sm_70 -j80
```

CLI guard:

```text
--mtp-serving commit --active-microbatch 2
rc=2
ds4-v100-replay: --mtp-serving commit currently requires --active-microbatch 1
```

Functional V100 validation used the source pack index because the disposable
build pod was recycled to clear stale defunct replay processes, which removed
the ephemeral `/workspace/ds4-appliance-full-tm-gated-s127` pack. This validates
the scheduler/MTP semantics but is not a production TurboMind throughput A/B.

| Case | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match | MTP attempted/accepted | Tensor-batched |
|---|---:|---:|---:|---:|---:|---:|
| 2-slot/256K verify, 2 requests x 2 tokens | `7.511943` | `0.834660` | `0.417330` | `2/2` | `2/2` | yes |
| 16-slot/256K verify, 16 requests x 2 tokens | `28.955967` | `3.217330` | `1.608665` | `16/16` | `16/16` | yes |

The 16-slot run showed `status_200=16`, `status_other=0`,
`tensor_batched_requests=16`, and `mtp_attempted=16`.

Decision: keep MTP verify as an explicit diagnostic/observability path, now
compatible with active microbatch serving. This does not change the practical
throughput target because verify still runs after base generation. True MTP
throughput still requires a speculative verifier that avoids serial target
recompute. Production-pack throughput should be rerun after restoring or
regenerating the TurboMind appliance pack in a persistent workspace.

## Files Summary

| File | Change |
|---|---|
| `ds4_v100_replay.h` | Add slot-specific output-HC read API |
| `ds4_v100_replay.c` | Implement slot-specific output-HC read |
| `tools/ds4-v100-replay.c` | Allow MTP verify batching and keep commit one-slot |
| `deploy/v100/ds4-v100-appliance.env.example` | Clarify verify versus commit batching |
| `docs/operations/DS4-V100-APPLIANCE.md` | Document MTP verify active microbatch behavior |
| `docs/sprints/VISION.md` | Record outcome |
| `logs/from-cluster/sprint180-mtp-verify-microbatch/` | V100 evidence |

## Definition Of Done

- [x] MTP verify active microbatch compiles on V100.
- [x] MTP commit active microbatch still fails closed.
- [x] Served MTP verify active microbatch returns per-response `mtp` JSON.
- [x] Served MTP verify records tensor-batch and MTP counters.
- [x] Prompt/prefill, generated, and continuation/decode tok/s are recorded.
- [x] Decision is recorded: promote only if it is useful operationally and does
      not regress base serving unacceptably; otherwise keep as explicit MTP
      diagnostics.
