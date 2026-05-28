# TEMP Status Report 452

## Current Focus

TP/EP operational-tier promotion for the router+FFN rank-major bundle.

## Result

Artifact:

- `/localpool/ds4/workspace/logs/s452-router-ffn-rankmajor-s28`

Shape:

- `slots=28`
- `ctx=262144`
- `requests=28`
- `tokens=4`
- `position=100000`

Both legs passed readiness. Response parity matched `28/28`.

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Control | 32.382224 | 32.177938 | 4.770488 | 11.323718% | 104539 | 2814 MiB |
| Router rank-major + FFN rank-major | 33.755509 | 33.718324 | 4.835465 | 11.836538% | 104539 | 2970 MiB |

Speedups:

- server generated decode: `1.0424x`
- server continuation decode: `1.0479x`
- client generated tok/s: `1.0136x`
- average GPU utilization: `1.0453x`
- HC-current input time: `0.8749x`
- min free VRAM: `1.0554x`

## Assessment

This is now promotion-worthy. It is not a large performance jump, but it is
clean at the practical 28-slot / 256K semantic tier, improves multiple
independent metrics, and keeps parity.

Next step: make router rank-major plus FFN rank-major a launcher/env default
with an opt-out, then run a default-vs-explicit-off A/B to verify the default
selection and preserve parity.

## Cluster State

After Sprint 452, the V100 node reported no active DS4 GPU jobs and all eight
GPUs had `0 MiB` used by DS4 processes.
