# Sprint 576 - C1 Full-Capture Is Batch-Unstable (Real Defect)

Date: 2026-05-29

## Goal

Determine whether the full-capture divergence is unavoidable numerical noise the
model tolerates, or a real bug, by measuring the divergence in logit space
instead of token space.

## Result summary

It is a **real full-capture defect**, not tolerated noise. The inherent serving
nondeterminism is tiny and discrete (identical eager runs are bit-identical on
matched tokens). Full-capture, under concurrency, is internally non-reproducible:
two identical full-capture runs diverge on every slot with large continuous logit
perturbations. This corrects Sprint 575's "per-slot correct => promotable" framing:
full capture is per-slot correct in isolation but batch-unstable, and is not
promotable as-is.

## Method

Added a per-step diagnostic to the output head (`engine/output_head.cu`): under
`--decode-stage-checksum-gate`, log per slot `tp_ep_decode_top1_logit` with
position, slot, selected token, and the global top-1 logit value. Rebuilt the
appliance. This measures the magnitude of perturbation in logit space, which
token-level comparison cannot (argmax amplifies any perturbation into a whole
token flip).

Ran three legs from the same binary with startup warmup disabled (so all legs
sit at the same decode positions, `250050-250081`), `SLOTS=32`, `32` identical
prompts, `32` tokens: `eager-A`, `eager-B`, `full`; then a separate pair
`full-A`, `full-B`. For each comparison, walked each slot's positions in order
and recorded `|logit_A - logit_B|` while the selected token matched (comparable
context), stopping at the first token flip.

## Results

| Comparison | matched-token \|logit Δ\| (max / mean / p50) | flips |
| --- | --- | ---: |
| `eager-A` vs `eager-B` (inherent floor) | `0` / `0` / `0` | `7/32` |
| `full` vs `eager-A` | `1.50` / `0.065` / `0` | `19/32` |
| `full-A` vs `full-B` (full's own floor) | `3.63` / `0.44` / `0.26` | `32/32` |

Logit scale for reference: values ~`14-27`, mean ~`18`.

## Interpretation

- **Inherent serving noise is minimal and discrete.** Two identical eager runs
  are bit-identical on matched tokens (`Δ = 0` to 9 significant figures). They
  differ only via `7/32` whole-token flips, concentrated at the first (most
  contended) decode step. That is the signature of MoE router top-k tie-breaking
  under reduction-order nondeterminism (`kernels/v100/router.cuh` atomics): when
  routing agrees the result is bit-exact; when a tie flips, the token routes to
  different experts and changes. This is the only inherent noise, and it is
  small.
- **Full capture is batch-unstable.** Two identical full-capture runs diverge on
  `32/32` slots with continuous logit deltas up to `3.63` (`p50 = 0.26`). A valid
  alternative arithmetic ordering would be reproducible run-to-run, like eager.
  This is non-reproducibility introduced by the graph-replay path, not tolerated
  noise. The ordering `eager-eager (0) < full-eager (1.5) < full-full (3.6)` is
  consistent with full capture being the unstable element (pairing two unstable
  runs maximizes divergence).
- **It is a batch/concurrency effect.** Sprint 575 showed single-slot full
  capture is bit-exact with eager over `32` tokens. So the instability appears
  only with concurrent slots: the captured/replayed graph interacts with the
  batched reductions (EP compose `atomicAdd`, route scratch, NCCL) in a way that
  is not deterministic across runs.

## Correction to Sprint 575

Sprint 575's single-slot bit-exactness is correct, but the inference "per-slot
correct => promotable" is wrong. Full capture is per-slot correct in isolation
and batch-unstable under concurrency. The divergences in Sprints 570-574 were
full capture's batch instability (large, real) conflated with the small discrete
eager routing-tie floor. The exact-equality oracle flagged the instability, but
the right conclusion is "fix the instability," not "the oracle is too strict."

## Decision

Full capture is **not promotable as-is**. There is a concrete defect to fix:
batch-level run-to-run instability in the graph-replay path.

Next (Sprint 577): localize the instability source with the logit diagnostic and
a deterministic-reduction probe. Leading hypotheses, in order:

1. **EP compose `atomicAdd`** (`kernels/v100/compose.cuh`) replayed inside the
   captured graph accumulating in a nondeterministic order that the eager path
   does not exhibit (eager-eager is bit-stable, so eager's accumulation is
   effectively deterministic for this shape; the graph changes that).
2. **Route/compose scratch not reset deterministically** between replays, so a
   replay reads residue from a prior step that varies run-to-run.
3. **HC rebase / buffer ping-pong** interacting with batched reductions.

Probe: rerun `full-A` vs `full-B` with a deterministic compose reduction (or
NCCL pinned to a deterministic algorithm and atomics replaced by a fixed-order
reduction in the captured region). If full's floor collapses toward the eager
floor, the source is confirmed.

## Definition of Done

- Logit diagnostic added and built; correctness-only.
- Floor measurements recorded (eager-eager, full-eager, full-full) at matched
  positions.
- Verdict recorded: real batch-instability defect, not tolerated noise.
- Sprint 575 correction recorded; steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`. The logit
  diagnostic in `engine/output_head.cu` is gated behind the existing
  `--decode-stage-checksum-gate` and is safe to keep.
