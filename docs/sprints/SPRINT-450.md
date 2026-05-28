# Sprint 450: Router Rank-Major Isolate Recheck

## Objective

Stay TP/EP only and run a clean HTTP isolate for
`--model-router-rank-major-logits-gate` on the rebuilt Sprint 448 binary.

The queued router-rank-major run that collided with Sprint 449 was interrupted,
so it is not a valid result. We need a clean isolate before deciding whether
router rank-major should be combined with routed-FFN rank-major.

## Implementation Plan

1. Use the rebuilt `sm_70` binary from Sprint 448.
2. Run a same-binary HTTP A/B at the reduced isolation shape:

   ```text
   slots=8
   ctx=262144
   requests=4
   tokens=2
   position=100000
   ```

3. Control remains the known-clean semantic TP/EP path.
4. Candidate enables only:

   ```text
   --candidate-model-router-rank-major-logits
   ```

5. Use a fresh port base that does not overlap earlier artifacts.

## Validation

- Remote HTTP A/B completes both legs.
- Response parity artifacts are present.
- No stale DS4 GPU process remains after the run.

## Decision Rule

- If parity fails, keep router rank-major default-off and do not combine it.
- If parity passes and throughput improves, next sprint should test
  `router rank-major + routed FFN rank-major`.
- If parity passes but is flat/slower, deprioritize router rank-major and move
  to FFN-rank-major-only larger-shape promotion testing.

## Execution Notes

The profile harness maps `--model-router-rank-major-logits` to a runtime command
that also enables `--routed-ffn-rank-major-input-gate`. Therefore this sprint is
best interpreted as "router rank-major on top of FFN rank-major", not a pure
router-only binary-level isolate.

Artifact:

```text
/localpool/ds4/workspace/logs/s450-router-rankmajor-isolate
```

The A/B parent hit the known handoff stall after the control leg. Signaling only
the wrapper PID allowed the candidate leg to launch. Because the wrapper was no
longer alive to run final checks, readiness and response parity were run
manually from the produced control/candidate directories.

## Outcome

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token |
|---|---:|---:|---:|---:|---:|
| Control | 20.089611 | 19.902711 | 0.736184 | 9.776316% | 71302 |
| Router rank-major + FFN rank-major | 20.785811 | 20.741867 | 0.753021 | 10.083333% | 71302 |

Response parity matched `4/4`:

```text
matched_pairs=4
failed_pairs=0
match=true
```

Manual readiness checks returned nonzero for both legs only because the reduced
two-token run misses the strict `client_generated_tok_s >= 1` threshold. The
model-serving checks, response files, status files, GPU samples, typed KV,
compact MoE, and checksums were present.

## Decision

Router rank-major plus FFN rank-major is a positive reduced-shape candidate:
server generated decode improved by about `3.47%`, continuation decode by about
`4.22%`, client throughput by about `2.29%`, and response parity stayed clean.

Next sprint should test this positive bundle at a larger serving shape before
considering launcher promotion. Do not add attention rank-local input to that
promotion candidate yet; Sprint 449 showed the attention leg cancels the FFN
gain in the reduced HTTP run.
