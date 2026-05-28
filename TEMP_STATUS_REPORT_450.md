# TEMP Status Report 450

## Current Focus

TP/EP rank-major serving bundle promotion candidates.

## Implemented

- Added Sprint 450 plan for a clean router rank-major recheck.
- Ran the HTTP A/B at `8` slots / `256K` / `4` requests / `2` tokens.
- Manually ran readiness and response parity because the A/B parent was
  signaled after the control leg to work around the known handoff stall.

## Result

Artifact:

- `/localpool/ds4/workspace/logs/s450-router-rankmajor-isolate`

Important interpretation: `--model-router-rank-major-logits` currently causes
the launched binary to enable both:

- `--model-router-rank-major-logits-gate`
- `--routed-ffn-rank-major-input-gate`

So this is a router-rank-major + FFN-rank-major bundle, not a pure router-only
runtime isolate.

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token |
|---|---:|---:|---:|---:|---:|
| Control | 20.089611 | 19.902711 | 0.736184 | 9.776316% | 71302 |
| Router rank-major + FFN rank-major | 20.785811 | 20.741867 | 0.753021 | 10.083333% | 71302 |

Response parity:

- `matched_pairs=4`
- `failed_pairs=0`
- `match=true`

Speedups:

- server generated decode: `1.0347x`
- server continuation decode: `1.0422x`
- client generated tok/s: `1.0229x`

## Assessment

This is the best rank-major reduced-shape bundle from the latest sequence:

- Attention rank-local alone: correctness clean, slower.
- FFN rank-major alone: correctness clean, faster.
- Attention rank-local + FFN rank-major: correctness clean, slightly slower.
- Router rank-major + FFN rank-major: correctness clean, faster.

Next step: test router rank-major + FFN rank-major at a larger serving shape
before any launcher promotion. Keep attention rank-local out of the promotion
candidate until it produces a net positive in a larger bundle.

## Cluster State

After Sprint 450, the V100 node reported no active DS4 GPU jobs and all eight
GPUs had `0 MiB` used by DS4 processes.
