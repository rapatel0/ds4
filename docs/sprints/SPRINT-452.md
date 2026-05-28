# Sprint 452: Operational-Tier Router+FFN Rank-Major A/B

## Objective

Stay TP/EP only and test the Sprint 451 positive router+FFN rank-major bundle
at the practical operational semantic serving tier.

Candidate bundle:

```text
--model-router-rank-major-logits-gate
--routed-ffn-rank-major-input-gate
```

## Implementation Plan

1. Run a same-binary HTTP A/B at:

   ```text
   slots=28
   ctx=262144
   requests=28
   tokens=4
   position=100000
   ```

2. Control remains the known-clean semantic TP/EP path.
3. Candidate enables:

   ```text
   --candidate-model-router-rank-major-logits
   ```

4. Preserve response parity, readiness, timing, GPU-utilization, and VRAM
   artifacts.

## Validation

- Remote HTTP A/B completes both legs.
- Both legs pass readiness or have a concrete documented readiness reason.
- Response parity artifacts are present.
- No stale DS4 GPU process remains after the run.

## Decision Rule

- If parity fails, keep the bundle default-off and debug correctness.
- If parity passes and throughput improves, prepare launcher/env promotion.
- If parity passes but throughput regresses, keep diagnostic-only and shift to
  graph/launch-count work.

## Outcome

Artifact:

```text
/localpool/ds4/workspace/logs/s452-router-ffn-rankmajor-s28
```

Both legs passed readiness. Response parity matched `28/28`.

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Control | 32.382224 | 32.177938 | 4.770488 | 11.323718% | 104539 | 2814 MiB |
| Router rank-major + FFN rank-major | 33.755509 | 33.718324 | 4.835465 | 11.836538% | 104539 | 2970 MiB |

Response parity:

```text
matched_pairs=28
failed_pairs=0
match=true
```

Speedups:

| Metric | Speedup |
|---|---:|
| Server generated decode | 1.0424x |
| Server continuation decode | 1.0479x |
| Client generated tok/s | 1.0136x |
| GPU util avg | 1.0453x |
| HC-current input time | 0.8749x |
| Min free VRAM | 1.0554x |

## Decision

Promote the router+FFN rank-major bundle into launcher/env defaults, with a
clear opt-out. The gain is modest but consistent across 8-, 16-, and 28-slot
HTTP A/Bs, preserves response parity, improves VRAM margin, and attacks the
current HC-current/router staging cost without adding attention rank-local
overhead.

Next sprint should make the promotion mechanical in the launcher/profile/env
defaults and then run a short default-vs-explicit-off A/B to verify the default
path selects the promoted gates.
