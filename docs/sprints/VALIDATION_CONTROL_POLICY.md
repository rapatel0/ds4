# Validation Control Policy

Last updated: 2026-05-28

## Default Gate

**Tolerance is the universal default for all cross-run serving validation.**
Bit-exact is opt-in only: a sprint plan must name the bit-exact gate
explicitly *and* justify why bit-exact is valid for that specific comparison.
If a sprint plan omits the gate, read it as tolerance.

The default gate:

- selected-token agreement ≥ `0.99` vs control on the reference shape, AND
- generated-sequence agreement ≥ `0.99` vs control.
- max selected-logit relative error is advisory only — report it for
  situational awareness but do not gate on it.

Tool: `tools/ds4-v100-http-response-tolerance.py`. The strict parity tool
`tools/ds4-v100-http-response-parity.py` is reserved for opt-in bit-exact
checks.

## When Bit-Exact Is Valid

Use bit-exact only when **both** of these hold:

1. The compared executions share a single process (no separate run, no NCCL
   re-initialization), OR the compared artifact is static (file checksum,
   built-binary hash, fixed-input single-kernel output without collectives).
2. The asserted property is "I changed nothing semantically" — a mechanical
   refactor, a kernel move from inline to a header, a build-only change.

If either does not hold, use tolerance.

Cross-run same-binary "reproducibility" checks are structurally invalid:
NCCL non-determinism makes two runs of the same binary against the same
input diverge at the ulp level, and autoregressive argmax amplifies the
divergence into token flips. **Do not schedule a two-run same-binary
sanity check.**

## Control Selection

For any cross-run comparison: **never run a fresh control by default.** The
current sprint's control is the most recent serving artifact that
represents the currently-promoted state. Two cases, both symmetric:

**Promote.** Last sprint's candidate won; it became the promoted state.
This sprint's control is that candidate's artifact.

```
sprint N-1:  control C(N-1)  ←  candidate B(N-1)  → PROMOTED
sprint N:    C(N) := B(N-1)     ← reuse the artifact, no fresh run
             B(N) → new candidate (the only run this sprint produces)
```

**Reject.** Last sprint's candidate failed; the promoted state is
unchanged. This sprint's control is the same artifact as last sprint's
control.

```
sprint N-1:  control C(N-1)  ←  candidate B(N-1)  → REJECTED
sprint N:    C(N) := C(N-1)     ← reuse the artifact, no fresh run
             B(N) → new candidate (the only run this sprint produces)
```

In both cases the sprint produces **exactly one new reference-shape run** —
the candidate. The control side is a pointer to an existing
`/localpool/ds4/workspace/<run-id>` artifact from a prior sprint.

A fresh control run is required only when one of these invalidates the
prior promoted artifact as a baseline:

- model artifacts changed (weights, scales, vocab),
- launcher defaults or environment changed,
- promoted flag set changed,
- serving shape changed,
- validation harness semantics changed.

If a sprint plan schedules a fresh control run, it must name which of those
five changed.

## Records

The canonical sprint record is `docs/sprints/SPRINT-NNN.md` plus the git
commit history. Per sprint, record in the sprint doc:

- The gate used (tolerance by default; "bit-exact opt-in — justification: …"
  otherwise).
- The control-side artifact path (`/localpool/ds4/workspace/<run-id>`) and
  whether it was reused from a prior sprint or freshly produced (and if
  fresh, which of the five invalidators required it).
- The candidate-side artifact path.
- The validation metrics for the gate used (agreement %, or the static-check
  result).

Per-sprint `TEMP_STATUS_REPORT_*.md` files at the repo root are retired.
Do not create them. Sprint outcomes belong in the sprint doc; run-level
data belongs in the workspace artifact directory.

## Sprint 493 Evidence

Sprint 493 attempted a duplicate same-binary two-run check at `32` slots /
`256K` context / `256` selected-token requests / `64` generated tokens.
Both legs served cleanly with:

- HTTP 200: `256/256`,
- `vram_failures=0`,
- NCCL graph SYS edges: `0`,
- `peer_copy_sys_ops=0`,
- `peer_copy_sys_bytes=0`.

The response-artifact parity check matched `32/256` pairs. Field-level
counts:

- selected-token matches: `189/256`,
- generated-sequence matches: `189/256`,
- decode-step checksum matches: `256/256`,
- final checksum matches: `32/256`.

That same-binary two-run pattern is structurally invalid: it spends a full
extra serving run and can still fail on cross-run drift even when both legs
are the same rebuilt binary and all serving/transport invariants pass.
This is the policy's founding evidence.

## Operational Rule

Sprint plans state, per validation step:

- The gate (tolerance / bit-exact opt-in with justification).
- The control artifact reference (reused from sprint N-K, or fresh with one
  of the five invalidators named).
- The candidate run to be produced (one per sprint).

A sprint plan that does not state these defaults to: tolerance gate,
control reused from prior promoted artifact, one candidate run.
