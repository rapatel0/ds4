# TEMP Status Report 395: Route-Plan Async Upload

Date: 2026-05-26

## Topline

Sprint 395 completed a default-off route-plan async upload gate, validated it
on the V100 pod, and promoted it to the TP/EP launcher/profile default.

Current best target-shape result from this sprint:

```text
32 requests / 32 slots / 256K / position 262080 / 32 generated tokens
model-router routes / compact MoE / prompt-file soak / VRAM report

server decode:       107.092211 tok/s
client generated:     37.239503 tok/s
avg GPU util:          9.295000%
route upload:          4.736281 ms
router D2H:            0.562918 ms
VRAM failures:         0
min free VRAM:      1746 MiB
```

## What Changed

- Added `--route-plan-async-upload-gate`.
- Added `DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD`, now defaulting to `1` in
  `tools/ds4-v100-run-appliance.sh`.
- Added persistent pinned host buffers for router selected experts, weights,
  offsets, route slots, route weights, compact route indices/counts, and the
  packed compact plan.
- Uploads route metadata to each destination rank with `cudaMemcpyAsync` on
  the rank stream.
- Added per-rank upload completion events so pinned buffers are not reused
  while a previous upload is still in flight.
- Added `--disable-route-plan-async-upload` to the profile harness for future
  same-binary control runs.

## A/B Result

| Metric | Control | Async upload |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| Response parity | `32/32` | `32/32` |
| Summary first token | `83484` | `83484` |
| Server decode tok/s | `104.834948` | `107.092211` |
| Client generated tok/s | `37.153198` | `37.239503` |
| Avg GPU util | `9.007813%` | `9.295000%` |
| Max GPU util | `50%` | `52%` |
| Route upload ms | `6.785109` | `4.736281` |
| Router D2H ms | `1.016605` | `0.562918` |
| HC-current FFN/router ms | `36.382578` | `33.878761` |
| Scaffold decode ms | `295.771385` | `290.432052` |
| Projected slot-step tok/s | `108.191670` | `110.180677` |
| Compressed-KV sum ms | `3419.366097` | `3407.598438` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

Permanent validators:

```text
response parity: match=true, matched_pairs=32, failed_pairs=0
candidate readiness: ready=true, failure_count=0
```

## Decision

Promote. This is not a large topline shift, but it is correct, reduces the
intended route upload boundary by about `30%`, improves server decode by about
`2.15%`, and does not change VRAM admission.

## Next Focus

Return to NCCL/collective work. The route-plan cleanup does not change the
larger diagnosis: the TP/EP path is still around `9-10%` average GPU
utilization at the target shape, so the next material question is whether the
current peer-copy collective transport should be replaced or augmented with an
NCCL-backed collective path.

Artifacts:

- `logs/from-cluster/sprint395-route-plan-async/`
