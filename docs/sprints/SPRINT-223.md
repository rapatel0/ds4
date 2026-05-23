# Sprint 223 - MTP Acceptance Matrix Pivot Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 222 proved the missing draft-side primitive: the MTP helper can expose
next-HC, chain a short MTP draft block, and verify that block through the
Sprint 221 target verifier. The first production-pack fixture accepted only
one token from a four-token draft block, so optimizing the verifier would be
premature.

Sprint 223 turns that single fixture into a repeatable acceptance matrix. The
goal is not to make MTP faster in this sprint. The goal is to decide, with
cluster evidence, whether MTP has enough accepted-prefix length on real prompts
to justify another serving optimization sprint. If it does not, the practical
serving branch should stop spending time on MTP and pivot back to attention/KV
or persistent low-bit execution.

## Goals

- Add a reusable V100 matrix harness for `--mtp-draft-block-smoke`.
- Run multiple prompt fixtures and block lengths against the production
  appliance pack without changing production defaults.
- Record accepted prefix, target forwards, effective output tokens,
  speculative saves, MTP timing, and verify timing per case.
- Produce a summary table and a machine-readable artifact that makes the
  decision obvious.
- Update docs with a clear continue/pivot decision for MTP.

## Non-Goals

- No MTP production promotion.
- No verifier graph capture or target-block speed optimization.
- No multi-slot MTP commit.
- No TP/PP scheduler changes.
- No model quality claims beyond draft/target token agreement.

## Implementation

1. Add `tools/ds4-v100-mtp-acceptance-matrix.sh`.
   - Accepts `--appliance-dir`, `--model`, `--mtp-model`, `--ctx`,
     `--prompts`, `--block-sizes`, `--tokens`, `--log-dir`, and optional
     `--expected-token-hex`.
   - Runs `tools/ds4-v100-replay --mtp-draft-block-smoke N --json` for each
     prompt/block-size case.
   - Writes per-case stdout/stderr logs.
   - Writes `mtp_acceptance_matrix.tsv`.
   - Writes `mtp_acceptance_summary.md`.
   - Fails closed if any case fails unexpectedly.
2. Keep the harness shell-only and explicit. It should not introduce a generic
   MTP scheduler abstraction.
3. Validate locally with `--help`/argument checks and shell syntax.
4. Copy the harness to the V100 pod and run the matrix on the production pack.
   Initial matrix:
   - prompts:
     - `short_reasoning_plain.txt`
     - `short_code_completion.txt`
     - `short_italian_fact.txt`
     - `long_code_audit.txt`
     - `long_memory_archive.txt`
   - block sizes: `2,4,8`
   - context: `262144`
   - slots/active microbatch: one-slot diagnostic path
5. Copy logs to
   `logs/from-cluster/sprint223-mtp-acceptance-matrix/`.
6. Update sprint/status/vision/runbook docs with the decision.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-mtp-acceptance-matrix.sh` | reusable acceptance matrix harness |
| `tools/ds4-v100-replay.c` | real-prompt compressed-cache cap sizing for diagnostics |
| `docs/sprints/SPRINT-223.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | vision update |
| `docs/operations/DS4-V100-APPLIANCE.md` | MTP diagnostic/runbook note |
| `logs/from-cluster/sprint223-mtp-acceptance-matrix/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Matrix harness exists and has documented usage.
- [x] Harness validates required inputs and block sizes.
- [x] Harness writes per-case logs, TSV, and Markdown summary.
- [x] Local shell validation passes.
- [x] V100 matrix runs on the production appliance pack.
- [x] Logs are copied to
      `logs/from-cluster/sprint223-mtp-acceptance-matrix/`.
- [x] Docs state whether MTP accepted-prefix evidence justifies another MTP
      optimization sprint or requires a pivot.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
bash -n tools/ds4-v100-mtp-acceptance-matrix.sh
tools/ds4-v100-mtp-acceptance-matrix.sh --help
git diff --check
make -B -j8 tools/ds4-v100-replay.o
```

V100 build:

```text
cd /workspace/ds4-sprint181
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

The first matrix attempt found a real diagnostic-capacity bug for long prompts:

```text
ds4-v100-replay: layer 2 decode failed: decode cache attention compressed capacity exceeded
```

Sprint 223 fixed that by sizing real-prompt replay compressed-cache caps from
prompt text length, matching the earlier synthetic-prompt fix from Sprint 185.
The full matrix then passed:

```text
cases: 15
ok_cases: 15
failed_cases: 0
average_accepted_prefix: 1.533
max_accepted_prefix: 2
cases_with_accepted_prefix_ge_2: 10
total_speculative_saves: 4
decision: continue-mtp-evaluation
```

Block-size detail:

| Block | Cases | Accepted Prefix >= 2 |
|---:|---:|---:|
| 2 | 5 | 4 |
| 4 | 5 | 3 |
| 8 | 5 | 3 |

Evidence is stored in
`logs/from-cluster/sprint223-mtp-acceptance-matrix/`.

## Decision

Continue MTP evaluation, but narrow it to a block-2 speculative serving path.

The useful signal is not long draft blocks. Across the 15-case matrix, accepted
prefix never exceeded `2`; block sizes `4` and `8` mostly add verifier work
without increasing accepted output. Block size `2` is the only plausible
throughput shape now: `4/5` block-2 cases accepted both drafted tokens and
reported `speculative_saves=1`, meaning two target forwards can potentially
advance three output tokens if the accepted prefix is committed correctly.

The next sprint should build a block-2 exact speculative commit/verify path
that preserves the current rollback contract and measures real continuation
throughput. Do not optimize block-4 or block-8 MTP until acceptance improves.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
model: /models/DSv4-Flash-256e-fixed.gguf
mtp_model: /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Matrix command:

```text
./tools/ds4-v100-mtp-acceptance-matrix.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --ctx 262144 \
  --prompts tests/test-vectors/prompts/short_reasoning_plain.txt,tests/test-vectors/prompts/short_code_completion.txt,tests/test-vectors/prompts/short_italian_fact.txt,tests/test-vectors/prompts/long_code_audit.txt,tests/test-vectors/prompts/long_memory_archive.txt \
  --block-sizes 2,4,8 \
  --tokens 16 \
  --log-dir /workspace/logs/sprint223-mtp-acceptance-matrix
```

Decision gate:

- Continue MTP optimization only if enough cases show non-trivial accepted
  prefixes, with special weight on `accepted_prefix_len >= 2` for block sizes
  `4` and `8`.
- Pivot away from MTP if most cases accept only `0-1` draft tokens and
  `speculative_saves` remains `0`.

## Risks

- Each case currently opens the production appliance process independently, so
  the full matrix may take time. That cost is acceptable for a decision gate.
- `--expected-token-hex 3136` is not valid for all prompts. The matrix should
  support expected-token checking but leave it off by default for multi-prompt
  sweeps.
- Low acceptance on the small prompt set is not a model-quality claim; it is a
  serving-engineering decision about whether to spend the next sprint on MTP.

## Security

Do not log model weights or full logits. Prompt fixtures are repo-local test
vectors.

## Dependencies

- Sprint 221 target block verifier.
- Sprint 222 MTP draft block smoke.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- MTP sidecar `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
