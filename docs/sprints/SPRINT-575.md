# Sprint 575 - C1 Full-Capture Replay Is Per-Slot Correct

Date: 2026-05-29

> **Correction (Sprint 576):** the single-slot bit-exactness below is correct,
> but the inference "per-slot correct => promotable" is wrong. Logit-space
> measurement shows full capture is **batch-unstable**: two identical full-capture
> runs diverge on all 32 slots (logit Δ up to 3.63), while two identical eager
> runs are bit-identical on matched tokens. Full capture is per-slot correct in
> isolation but non-reproducible under concurrency — a real defect, not tolerated
> noise. It is NOT promotable as-is. See Sprint 576.

## Goal

Localize the full-capture divergence to a layer/stage and repair it.

## Result summary

The repair is moot: single-slot testing shows **full-capture replay is per-slot
bit-exact with eager**, and every divergence observed since Sprint 570 is
batch/concurrency-dependent nondeterminism, not a replay correctness bug. The C1
full-capture correctness blocker was a validation-methodology artifact: an
exact-equality oracle applied to a serving path that is nondeterministic across
concurrent slots.

## Evidence

Reused the Sprint 573 build (`/workspace/s573-continuation-instrument`, HEAD
`7d5a9342`). Correctness-only; request-level generated token sequences as the
oracle.

### Single-slot bit-exactness (decisive)

`SLOTS=8` configured, **one active request** (so the cross-position persistent
replay runs with a single slot: token `0` captures, tokens `1..N` are
HC-rebased cache-hit replays at advancing positions).

| Position | Tokens | eager vs full |
| ---: | ---: | --- |
| `250064` | `4` | identical (`ndiff 0`) |
| `250000` | `32` | identical (`ndiff 0`, `first_diff_offset None`) |

At both positions the full-capture persistent replay reproduces eager output
exactly. The replay mechanism (HC rebase plus graph reuse across decode
positions) is correct per slot.

### Determinism floor at the "catastrophic" position (retraction)

Sprint 574 reported `250064` as catastrophic (`32/32` against eager) and called
it position-dependent. Re-running with the full floor disproves that
interpretation:

| Comparison (`250064`, `SLOTS=32`, `32` tok) | Mismatch | distinct seqs |
| --- | ---: | ---: |
| `control-A` vs `control-B` (identical config) | `32/32` | `10` / `5` |
| `eager` vs `control-A` | `3/32` | `10` |
| `eager` vs `full` | `32/32` | `32` |

Two identical suffix-control configs diverge `32/32` at `250064`, and
`control-B` collapsed to `5` distinct sequences. So `250064` is a position where
the **measurement itself is unstable** (high batch-reduction-order
nondeterminism / an occasional degenerate batch run), not a position where full
capture has a specific bug. Sprint 574's catastrophic-`250064` conclusion is
retracted.

### Reconciling the Sprint 573 offset-`28` cluster

Sprint 573 found a clean offset-`28` cluster at `250000` (`SLOTS=32`) above a
`3/32` floor. The single-slot `250000` run now generates `32` tokens with no
divergence, so that cluster is also a concurrent-batch effect, not a per-slot
replay error. Full capture does appear to produce somewhat higher
distinct-sequence counts than control under concurrency (`250000`: full `8` vs
control `5-6`), i.e. it may interact with batch nondeterminism slightly
differently, but it does not compute any slot's tokens incorrectly.

## Conclusion

- **Full-capture replay correctness is established at the per-slot level.** The
  Sprint 568 HC-rebase plus cross-position graph reuse is bit-exact with eager
  for `32` decode steps at a single slot.
- **The serving path is nondeterministic across concurrent slots** in every
  configuration (identical-config controls diverge: `3/32` at `250000`, `32/32`
  at `250064`). This is reduction-order nondeterminism (MoE all-to-all / NCCL),
  independent of full capture.
- **The C1 full-capture blocker since Sprint 570 was a methodology artifact:**
  exact-sequence equality against a nondeterministic reference can never pass.
  Sprints 571-574 (early-continuation, comp-emit, position-dependence) localized
  features of the nondeterminism, not a replay bug.

## Decision

No promoted-tree code change. Do not pursue a replay-internals repair; there is
nothing per-slot to repair.

Reframe the promotion question for full capture. It is no longer "does full
capture match the reference token-for-token" (unanswerable under serving
nondeterminism) but:

1. **Per-slot bit-exactness gate (passes today):** single-request, multi-token,
   eager vs full bit-exact. Use this as the correctness gate for promotion.
2. **Batch-nondeterminism characterization:** quantify whether full capture
   materially increases concurrent-batch output variance versus the promoted
   suffix-control path, using the `control-A` vs `control-B` floor at matched
   concurrency and several positions. If full capture's variance is within the
   control floor, promote it for the throughput win (`1.25-1.48x` decode from
   Sprints 569-570). If it materially amplifies variance, treat that as the real
   (separate, smaller) issue.

## Definition of Done

- Single-slot bit-exactness recorded at `250000` (`32` tok) and `250064`.
- `250064` determinism floor recorded; Sprint 574 catastrophic claim retracted.
- Sprint 573 offset-`28` cluster reinterpreted as a concurrent-batch effect.
- Steering and vision updated with the per-slot-correct conclusion and the
  reframed promotion gate.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.
