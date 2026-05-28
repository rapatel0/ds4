# TEMP Status Report 451

## Current Focus

TP/EP rank-major bundle promotion testing at a larger serving shape.

## Result

Artifact:

- `/localpool/ds4/workspace/logs/s451-router-ffn-rankmajor-s16`

Shape:

- `slots=16`
- `ctx=262144`
- `requests=16`
- `tokens=4`
- `position=100000`

Both legs passed readiness. Response parity matched `16/16`.

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Control | 27.178499 | 27.084010 | 3.786283 | 10.560345% | 128816 | 4590 MiB |
| Router rank-major + FFN rank-major | 28.116301 | 28.121178 | 3.891491 | 11.758621% | 128816 | 4742 MiB |

Speedups:

- server generated decode: `1.0345x`
- server continuation decode: `1.0383x`
- client generated tok/s: `1.0278x`
- average GPU utilization: `1.1135x`
- HC-current input time: `0.9283x`

## Assessment

The router+FFN rank-major bundle survived the larger-shape check. It is not a
huge win, but it is correctness-clean and directionally improves server decode,
client throughput, GPU utilization, and VRAM margin.

Next step is a target-shape check at `28` or `32` slots / `256K` before changing
launcher defaults. Keep attention rank-local out of the candidate for now
because Sprint 449 showed it cancels the rank-major win in the reduced run.

## Cluster State

After Sprint 451, the V100 node reported no active DS4 GPU jobs and all eight
GPUs had `0 MiB` used by DS4 processes.
