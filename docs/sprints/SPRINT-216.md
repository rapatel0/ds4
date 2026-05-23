# Sprint 216 - True MTP Speculative Verification Gate

Date: 2026-05-23
Status: Planned

## Overview

Sprint 215 showed that the current deployable appliance tops out at
`68.403129` continuation tok/s for `32` slots at `128K`, while the maximum
context production mode remains `16` slots at `256K` with `61.624766`
continuation tok/s. It also showed that current MTP modes are operational but
not throughput features: verify accepted `0/16` drafts, and one-slot commit
accepted `8/15` drafts while still running slowly because it does not avoid
target forwards.

Sprint 216 is the gate for real speculative decoding. The question is narrow:
can the current replay/runtime stack verify drafted tokens in a way that
reduces target-model work and advances state correctly, or does it only support
serial one-token target replay today?

## Goals

- Build a focused MTP speculative-verification harness around the current
  replay/runtime APIs.
- Report target forwards separately from accepted or committed draft tokens.
- Prove whether accepted drafts actually reduce target-model forwards.
- Record the exact runtime/API gap if multi-token target verification cannot be
  implemented without a broader scheduler change.
- Keep MTP default-off and correctness-gated.

## Non-Goals

- No generic scheduler abstraction.
- No TP/PP scheduler integration.
- No routed-FFN kernel tuning.
- No production MTP default change.
- No speedup claim based only on accepted-draft counters.
- No fake effective tok/s that counts committed drafts while still performing
  the same target-model work.

## Implementation

1. Audit the current MTP commit and verify paths in:

| Area | Files |
|---|---|
| replay API | `ds4_v100_replay.c`, `ds4_v100_replay.h` |
| replay CLI/server | `tools/ds4-v100-replay.c` |
| MTP sidecar | `ds4_v100_mtp.c` |
| benchmark wrappers | `tools/ds4-v100-sustained-decode-bench.sh`, `tools/ds4-v100-practical-serving-matrix.sh` |

2. Add focused reporting for speculative-decode accounting:

| Metric | Meaning |
|---|---|
| `draft_tokens_proposed` | sidecar draft candidates attempted |
| `draft_tokens_accepted` | draft candidates matching target verification |
| `accepted_prefix_len` | accepted speculative prefix length for a target verification step |
| `target_tokens_verified` | target logits checked against proposed draft tokens |
| `target_forwards` | number of target-model forwards actually performed |
| `effective_output_tokens` | user-visible tokens emitted |
| `speculative_saves` | avoided target forwards, must be `>0` to claim speedup |

3. If the current code only supports serial target verification, expose that
   explicitly in the MTP commit result and keep `speculative_saves=0`.
4. If a block verification primitive already exists or can be added narrowly,
   implement a harness-only path that verifies a drafted block and compares the
   emitted sequence against normal one-slot generation.
5. Run the focused gate on the V100 pod against the production pack and MTP
   sidecar at `256K`, one slot, with prompt/prefill and continuation metrics
   recorded separately.
6. Update status, vision, and the appliance runbook with the decision:
   integrate true MTP next, or pivot away because a scheduler/runtime primitive
   is missing.

## Parallel Work Lanes

| Lane | Work | Write scope |
|---|---|---|
| A | Replay/MTP API audit and accounting fields | `tools/ds4-v100-replay.c`, `ds4_v100_replay.*` |
| B | Focused MTP gate harness or benchmark wrapper | `tools/ds4-v100-*mtp*` or existing sustained bench wrapper |
| C | V100 build/run and evidence capture | `logs/from-cluster/sprint216-mtp-spec-gate/` |
| D | Decision docs and operator guidance | sprint/status/vision/runbook docs |

Workers are not alone in the codebase. Do not revert unrelated edits. Keep any
TP work in separate TP-only files; this sprint is MTP-only.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-replay.c` | MTP commit/verify accounting and focused gate path |
| `ds4_v100_replay.c` | replay primitive audit or narrow API addition |
| `ds4_v100_replay.h` | typed API update if a narrow primitive is added |
| `ds4_v100_mtp.c` | sidecar draft/verify audit only |
| `tools/ds4-v100-sustained-decode-bench.sh` | focused benchmark invocation/reporting |
| `docs/operations/DS4-V100-APPLIANCE.md` | operator MTP guidance |
| `docs/sprints/STATUS.md` | topline status |
| `docs/sprints/VISION.md` | vision progress and next lever |
| `logs/from-cluster/sprint216-mtp-spec-gate/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] Current MTP commit path reports target forwards and speculative saves.
- [ ] A focused MTP speculative-verification harness or mode exists.
- [ ] The harness reports draft proposals, accepted prefix, target tokens
      verified, target forwards, effective output tokens, and effective tok/s.
- [ ] If block target verification is impossible with current APIs, the exact
      missing primitive is documented with code references.
- [ ] If block target verification is possible, emitted tokens match normal
      one-slot replay for the same prompt and token count.
- [ ] V100 build passes for any C/CUDA changes.
- [ ] V100 focused run is captured under
      `logs/from-cluster/sprint216-mtp-spec-gate/`.
- [ ] Prompt/prefill and continuation/decode tok/s are recorded separately.
- [ ] MTP remains default-off unless the focused gate proves real target-forward
      savings and correctness.
- [ ] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Verification Strategy

Use the V100 build pod and persistent production pack:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
base model: /models/DSv4-Flash-256e-fixed.gguf
mtp model: /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Run a focused one-slot `256K` test first. If the accounting gate proves
`speculative_saves=0`, stop before broad serving integration and document the
required scheduler/runtime primitive. If the gate proves positive savings,
follow with a same-prompt one-slot A/B against normal generation.

## Decision Gates

Promote MTP to the next integration sprint only if:

- token correctness matches normal target generation;
- `target_forwards < effective_output_tokens` for the accepted sequence;
- effective continuation tok/s improves after accounting for draft overhead;
- state advancement is explicit rather than replaying accepted drafts serially.

Reject or defer MTP production integration if:

- accepted drafts are only counters and do not reduce target forwards;
- the replay API can only feed one position per target forward;
- KV/cache state for accepted drafts cannot be advanced without serial replay;
- MTP sidecar only proposes one token per target state and cannot build a useful
  verification block.

## Risks

- The replay API may only batch across slots, not multiple positions for one
  slot.
- Accepted drafts may require target KV/state updates that currently happen
  only through serial base replay.
- A narrow harness can prove feasibility but still leave a production scheduler
  integration sprint.
- One-slot MTP acceptance may not generalize to 16/32-slot production serving.

## Security

No external exposure. No model weights in logs. Use the existing cluster pod,
local model mounts, and persistent appliance pack.

## Dependencies

- Sprint 215 practical serving matrix and MTP counters.
- Production pack from Sprint 181:
  `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- V100 build pod `llm/llamacpp-build-8gpu`.
