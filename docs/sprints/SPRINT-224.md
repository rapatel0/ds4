# Sprint 224 - MTP Block-2 Exact Commit Throughput Gate

Date: 2026-05-23
Status: Complete - Gate Failed For Serving Integration

## Overview

Sprint 223 changed the MTP decision. MTP is not ready for production default,
but it is also not a dead end. The acceptance matrix shows one viable shape:
draft block size `2`. Four of five block-2 prompt fixtures accepted both draft
tokens, and those cases reported `speculative_saves=1`. Longer blocks did not
help; accepted prefix never exceeded `2`.

Sprint 224 should therefore stop testing arbitrary draft lengths and implement
a focused block-2 exact speculative commit path. The sprint must prove whether
the accepted-prefix signal can turn into real continuation tok/s. If it cannot,
MTP should pause again and practical serving should return to attention/KV or
persistent low-bit execution.

## Goals

- Add an opt-in block-2 MTP exact commit path that:
  - drafts two MTP tokens from the first target token and MTP next-HC;
  - verifies exactly those two forced tokens through the target block verifier;
  - commits only the accepted prefix;
  - restores/replays safely when fewer than two drafts are accepted;
  - reports target forwards, effective output tokens, accepted prefix, and
    speculative saves.
- Add a direct replay diagnostic/benchmark mode for the block-2 path.
- Run same-prompt V100 A/B against baseline one-slot `256K` direct replay.
- Decide whether block-2 MTP should be promoted to a guarded serving
  integration sprint or rejected as overhead-bound.

## Non-Goals

- No block-4 or block-8 optimization.
- No skip-verify MTP.
- No multi-slot MTP commit.
- No default production change.
- No TP/PP scheduler changes.

## Implementation

1. Refactor the Sprint 222 draft-block logic into reusable helpers where
   needed:
   - prompt replay to pre-draft state;
   - first target token selection;
   - target HC read;
   - two-step MTP draft chain with next-HC;
   - target block verify.
2. Add a guarded block-2 commit diagnostic mode, tentatively
   `--mtp-block2-commit-smoke N` or a similarly explicit name.
   - Requires `--mtp-model`, `--slots 1`, and `--active-microbatch 1`.
   - Runs up to `N` output tokens.
   - Emits JSON and text accounting.
   - Produces the same final token sequence as baseline greedy replay for the
     covered fixture.
3. For accepted prefix `2`, commit all three effective output tokens produced
   by two target forwards: first target token plus two accepted MTP drafts.
4. For accepted prefix `0` or `1`, restore to the snapshot and replay only the
   safe accepted prefix plus the target fallback path. Do not leave the target
   state advanced past a rejected draft.
5. Add harness support only if needed to run A/B cleanly; avoid introducing a
   generic MTP scheduler abstraction.
6. Build locally and on the V100 pod.
7. Run V100 direct replay A/B on production pack:
   - baseline off;
   - block-2 exact commit;
   - prompts from the Sprint 223 matrix, with special attention to the four
     block-2 accepting cases.
8. Copy logs to
   `logs/from-cluster/sprint224-mtp-block2-commit/`.
9. Update sprint/status/vision/runbook docs with the decision.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-replay.c` | block-2 exact commit diagnostic and accounting |
| `tools/ds4-v100-mtp-acceptance-matrix.sh` | optional reuse for prompt list/A-B helpers |
| `docs/sprints/SPRINT-224.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | vision update |
| `docs/operations/DS4-V100-APPLIANCE.md` | MTP runbook note |
| `logs/from-cluster/sprint224-mtp-block2-commit/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Block-2 exact commit diagnostic exists and fails closed outside one-slot
      MTP mode.
- [x] Accepted-prefix commit restores/replays safely when prefix is less than
      two.
- [ ] Output token sequence matches baseline for the measured prompts.
- [x] Accounting reports accepted prefix, target forwards, effective output
      tokens, speculative saves, draft time, verify time, and total timing.
- [x] Local validation passes.
- [x] V100 build passes.
- [x] V100 A/B runs on the production appliance pack.
- [x] Logs are copied to
      `logs/from-cluster/sprint224-mtp-block2-commit/`.
- [x] Docs state whether block-2 exact commit justifies a serving integration
      sprint.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
git diff --check
make -B -j8 tools/ds4-v100-replay.o
```

V100 build:

```text
cd /workspace/ds4-sprint181
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Negative guard:

```text
ds4-v100-replay: --mtp-block2-commit-smoke currently requires --slots 1 --active-microbatch 1
negative_rc=2
```

V100 production-pack block-2 matrix:

```text
cases: 5
ok_cases: 4
failed_cases: 1
average_block2_tps_ok_cases: 3.663043
average_baseline_tps_ok_cases: 2.032918
ok_case_speedup_ratio: 1.801865
```

| Case | Status | Match | Blocks | Full | Partial | Reject | Drafts Accepted | Target Forwards | Spec Saves | Block2 tok/s | Baseline tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_reasoning_plain` | ok | true | 4 | 0 | 2 | 2 | 2 | 7 | 1 | 5.543783 | 3.043298 |
| `short_code_completion` | ok | true | 4 | 3 | 0 | 1 | 6 | 7 | 1 | 4.105815 | 2.430585 |
| `short_italian_fact` | ok | true | 3 | 3 | 0 | 0 | 6 | 7 | 1 | 4.969561 | 2.624934 |
| `long_code_audit` | ok | true | 3 | 3 | 0 | 0 | 6 | 7 | 1 | 0.033014 | 0.032855 |
| `long_memory_archive` | fail | false | - | - | - | - | - | - | - | - | - |

The failed case is not isolated to block-2 MTP. `long_memory_archive` failed
token parity at token 1 (`baseline=16`, `got=8773`), and a follow-up
`--target-block-smoke 2` on the same prompt also failed reset determinism:

```text
ds4-v100-replay: target-block prompt[0] token mismatch got=32085 want=10220
ds4-v100-replay: target block prompt first token mismatch
```

Evidence is stored in
`logs/from-cluster/sprint224-mtp-block2-commit/`.

## Decision

Do not move block-2 MTP into serving integration yet.

The positive signal is real: the exact block-2 path is token-correct on four
of five prompts and faster than the same-process baseline on those successful
cases. It also performs the useful state transition: accepted prefix `2` emits
three output tokens with two target forwards, while partial/reject blocks keep
state aligned with the verified target token.

The blocker is correctness scope. The `long_memory_archive` fixture shows
target replay reset nondeterminism even without the block-2 path, so MTP cannot
be promoted safely until long-prompt replay reset/snapshot determinism is
fixed or the serving contract explicitly excludes that class of prompt. The
next sprint should address replay reset determinism for long prompts, then
rerun this block-2 gate.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
model: /models/DSv4-Flash-256e-fixed.gguf
mtp_model: /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Minimum A/B prompts:

```text
tests/test-vectors/prompts/short_code_completion.txt
tests/test-vectors/prompts/short_italian_fact.txt
tests/test-vectors/prompts/long_code_audit.txt
tests/test-vectors/prompts/long_memory_archive.txt
```

Decision gate:

- Continue to serving integration if block-2 exact commit preserves token
  sequence and improves effective continuation tok/s on the accepting prompts.
- Reject or pause MTP if the exact commit path is slower after including MTP
  draft and verification overhead, or if state replay for partial accepts is
  fragile.

## Risks

- Exact verification may still cost too much even when acceptance is good.
- Snapshot restore/replay for partial accepts can hide bugs if the benchmark
  only covers fully accepted prompts.
- The direct replay path can overstate serving gains if HTTP batching and
  queueing are not measured later. A positive direct gate should lead to a
  separate serving-integration sprint, not immediate default promotion.

## Security

No external serving exposure. Do not log model weights or full logits.

## Dependencies

- Sprint 221 target block verifier.
- Sprint 222 MTP next-HC draft chaining.
- Sprint 223 acceptance matrix.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- MTP sidecar `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
