# Sprint 451: Larger-Shape Router+FFN Rank-Major A/B

## Objective

Stay TP/EP only and test the Sprint 450 positive bundle at a larger serving
shape before considering launcher promotion.

Candidate bundle:

```text
--model-router-rank-major-logits-gate
--routed-ffn-rank-major-input-gate
```

The profile harness enables the FFN rank-major input gate when router
rank-major logits are requested, so the candidate is treated as a bundle.

## Implementation Plan

1. Run a same-binary HTTP A/B at:

   ```text
   slots=16
   ctx=262144
   requests=16
   tokens=4
   position=100000
   ```

2. Control remains the known-clean semantic TP/EP path.
3. Candidate enables:

   ```text
   --candidate-model-router-rank-major-logits
   ```

4. Use a fresh port base and preserve all response/readiness/parity artifacts.

## Validation

- Remote HTTP A/B completes both legs.
- Response parity artifacts are present.
- No stale DS4 GPU process remains after the run.

## Decision Rule

- If parity fails, keep the bundle default-off and debug before any promotion.
- If parity passes and throughput improves materially, promote to the next
  target-shape test.
- If parity passes but gains vanish, keep the bundle diagnostic-only and shift
  to graph/launch-count work.

## Outcome

Artifact:

```text
/localpool/ds4/workspace/logs/s451-router-ffn-rankmajor-s16
```

The A/B completed both legs without the parent handoff workaround. Both legs
passed readiness and response parity matched `16/16`.

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Control | 27.178499 | 27.084010 | 3.786283 | 10.560345% | 128816 | 4590 MiB |
| Router rank-major + FFN rank-major | 28.116301 | 28.121178 | 3.891491 | 11.758621% | 128816 | 4742 MiB |

Response parity:

```text
matched_pairs=16
failed_pairs=0
match=true
```

Speedups:

| Metric | Speedup |
|---|---:|
| Server generated decode | 1.0345x |
| Server continuation decode | 1.0383x |
| Client generated tok/s | 1.0278x |
| GPU util avg | 1.1135x |
| HC-current input time | 0.9283x |

## Decision

Promote the router+FFN rank-major bundle to the next target-shape test. This is
the first rank-major bundle in the current sequence that is correctness-clean
and remains positive beyond the tiny two-token isolate.

Do not yet make it a launcher default. The next sprint should test the same
candidate at a more operational shape, preferably `28` or `32` slots at
`256K`, before changing default serving configuration.
