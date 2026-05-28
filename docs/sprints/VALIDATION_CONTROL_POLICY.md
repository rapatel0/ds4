# Validation Control Policy

Last updated: 2026-05-28

## Control Selection

Use the latest promoted serving run as the control leg when all of these are
unchanged:

- model artifacts,
- launcher defaults and environment,
- promoted flag set,
- serving shape,
- validation harness semantics.

Do not schedule a fresh control run by default. A new control run is required
only when one of those inputs changed or when the existing promoted-control
artifact does not cover the evidence needed for the decision.

## Gate Selection

Use the tolerance gate for cross-run serving comparisons unless a sprint plan
explicitly requests bit-exact validation and explains why bit-exact is valid for
that comparison.

Use bit-exact checks for same-process/static checks where the compared value is
expected to be identical, such as fixed-input unit kernels, build artifacts, or
single-process A/B checks that do not reinitialize collectives.

## Sprint 493 Evidence

Sprint 493 attempted a duplicate same-binary two-run check at `32` slots /
`256K` context / `256` selected-token requests / `64` generated tokens. Both
legs served cleanly with:

- HTTP 200: `256/256`,
- `vram_failures=0`,
- NCCL graph SYS edges: `0`,
- `peer_copy_sys_ops=0`,
- `peer_copy_sys_bytes=0`.

The response-artifact parity check matched `32/256` pairs. Field-level counts
were:

- selected-token matches: `189/256`,
- generated-sequence matches: `189/256`,
- decode-step checksum matches: `256/256`,
- final checksum matches: `32/256`.

That run is not useful as a default control strategy: it spends a full extra
serving run and can still fail on cross-run drift even when both legs are the
same rebuilt binary and all serving/transport invariants pass.

## Operational Rule

For phase-boundary validation, record the control artifact or promoted-control
lineage used. If only the changed candidate is run, record why the previous
promoted control still applies.
