# TEMP Status Report 413

Date: 2026-05-26

## Current Focus

Sprint 413 tested reduced-slot TP/EP semantic serving at the required `256K`
context. This followed the user steer that slots can be reduced to improve
practical operation.

## What Changed

- Relaxed `tools/ds4-v100-run-appliance.sh` so TP/EP serving accepts
  `DS4_V100_SLOTS<=32` instead of requiring exactly `32`.
- Kept `32` as the long-term target.
- Kept active microbatch equal to configured slots for this serving path.
- Ran real V100 HTTP A/B tests with HC-current NCCL, lazy output head, compact
  MoE, post-attention FFN input, and attention-output NCCL on the candidate.

## Results

| Slots | Candidate HTTP | Ready | Server decode tok/s | Client generated tok/s | Min free VRAM | Post-close free | VRAM failures |
|---:|---:|---|---:|---:|---:|---:|---:|
| 24 | `24/24` | `true` | `19.716583` | `7.367808881719989` | `2428 MiB` | `2562 MiB` | `0` |
| 28 | `28/28` | `true` | `20.624419` | `7.922564755228921` | `1790 MiB` | `1924 MiB` | `0` |
| 30 | `30/30` | `true` | `21.089170` | `8.21212433262282` | `1556 MiB` | `1692 MiB` | `0` |

Prior `32` slot semantic serving from Sprint 412 served `32/32` responses but
failed readiness with `1328 MiB` minimum free VRAM and `62` reserve failures.

## Decision

- Highest clean tier tested: `30` slots.
- Practical tier for follow-on semantic serving work: `28` slots.
- Reason: `30` passes by only `20 MiB` above the `1536 MiB` NCCL reserve, while
  `28` leaves `254 MiB` above the reserve.

## Bottleneck

This sprint made the semantic path operational at reduced slots; it did not
solve throughput. The candidate remains roughly `5x` slower than the fast
control. The measured semantic overhead is concentrated in:

- true attention-output projection/allgather
- post-attention FFN input materialization
- downstream EP work after the semantic tensor is enabled

## Artifacts

```text
logs/from-cluster/sprint413-post-attn-slot24-http-ab-rerun/
logs/from-cluster/sprint413-post-attn-slot28-http-ab/
logs/from-cluster/sprint413-post-attn-slot30-http-ab/
```

## Next

Use `28` slots / `256K` as the practical semantic-serving benchmark shape and
optimize the true attention-output/post-attention path. Return to `32` slots
after the semantic path has at least several hundred MiB of additional reserve
headroom.
