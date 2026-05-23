# Sprint 216 - True MTP Speculative Verification Gate

Date: 2026-05-23
Status: Complete

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

- [x] Sprint plan exists and is committed before execution evidence is staged.
- [x] Current MTP commit path reports target forwards and speculative saves.
- [x] A focused MTP speculative-verification harness or mode exists.
- [x] The harness reports draft proposals, accepted prefix, target tokens
      verified, target forwards, effective output tokens, and effective tok/s.
- [x] If block target verification is impossible with current APIs, the exact
      missing primitive is documented with code references.
- [x] If block target verification is possible, emitted tokens match normal
      one-slot replay for the same prompt and token count.
- [x] V100 build passes for any C/CUDA changes.
- [x] V100 focused run is captured under
      `logs/from-cluster/sprint216-mtp-spec-gate/`.
- [x] Prompt/prefill and continuation/decode tok/s are recorded separately.
- [x] MTP remains default-off unless the focused gate proves real target-forward
      savings and correctness.
- [x] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

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

## Execution

Implemented honest speculative-accounting fields in
`tools/ds4-v100-replay.c`:

| Field | Purpose |
|---|---|
| `draft_tokens_proposed` | number of sidecar draft attempts |
| `draft_tokens_accepted` | number of draft tokens matching target |
| `accepted_prefix_len` | maximum accepted prefix length visible to current path |
| `target_tokens_verified` | target tokens compared against drafts |
| `target_forwards` | target-model output steps actually performed |
| `effective_output_tokens` | user-visible output tokens |
| `speculative_saves` | avoided target forwards |

Extended `tools/ds4-v100-sustained-decode-bench.sh` so MTP summaries aggregate
those fields, and added `tools/ds4-v100-mtp-spec-gate.sh` as the focused gate
harness. The harness runs:

1. one-slot `256K` target generation with MTP off;
2. one-slot `256K` MTP commit mode;
3. a small JSON/Markdown report comparing target forwards, effective output
   tokens, and speculative saves.

## V100 Evidence

Cluster target: `llm/llamacpp-build-8gpu` on `gpu-01`.

Build:

```text
cd /workspace/ds4-sprint181
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Focused gate command:

```text
cd /workspace/ds4-sprint181
./tools/ds4-v100-mtp-spec-gate.sh \
  --log-dir /workspace/logs/sprint216-mtp-spec-gate \
  --port-base 19040 \
  --requests 1 \
  --warmup-requests 0 \
  --tokens 16 \
  --ctx 262144
```

Topline:

| Mode | Generated tok/s | Continuation tok/s | Match | Drafts accepted | Target forwards | Effective output tokens | Spec saves |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline off | `4.954613` | `4.644949` | 1/1 | n/a | n/a | n/a | n/a |
| MTP commit | `4.561292` | `4.276211` | 1/1 | `8/15` | `16` | `16` | `0` |

Evidence:

```text
logs/from-cluster/sprint216-mtp-spec-gate/
```

## API Gap

The current replay API batches across slots, not speculative positions for one
slot. `ds4_v100_replay_generate_batch()` builds `batch_tokens[]` and
`batch_positions[]` with one entry per active slot, then advances all slots one
decode step at a time. There is no public primitive that accepts a drafted
token block for a single slot and returns target logits for all drafted
positions in one verification pass.

Relevant code:

- `ds4_v100_replay.h:122` exposes `ds4_v100_replay_generate_batch()` with
  `n_prompts`, not a one-slot multi-position verification block.
- `ds4_v100_replay.h:148` exposes `ds4_v100_replay_feed_token_at_position()`,
  a single-token/single-position feed primitive.
- `ds4_v100_replay.c:2498` through `2627` implements batch generation by
  iterating `step` and feeding one token per slot at each step.
- `tools/ds4-v100-replay.c:1307` through `1428` implements MTP commit by
  feeding the committed token, selecting the target token, and then running the
  MTP sidecar draft comparison serially for each generated step.

The smallest missing primitive is a target-model speculative verification API
that can:

1. take one slot's current KV/state plus a drafted token block;
2. advance target KV/state over the block in one scheduled verification unit;
3. return target logits for each drafted position;
4. commit only the accepted prefix or roll back rejected suffix state without
   serially replaying the same target forwards.

## Decision

Sprint 216 fails the true speculative MTP gate for the current implementation.
MTP commit accepts drafts, but accepted drafts do not save target-model work:
`target_forwards=16`, `effective_output_tokens=16`, and `speculative_saves=0`.

Keep `DS4_V100_MTP_SERVING=off` for production throughput serving. MTP verify
and commit remain diagnostics until a new target-verification/state-advance
primitive exists. The next practical-serving sprint should either build that
primitive explicitly or pivot to the 256K attention/KV execution boundary.
