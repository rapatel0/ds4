# Sprint 395: Async Route-Plan Upload Gate

## Overview

Add a default-off TP/EP route-planning upload gate for the real-router
compact-MoE serving path.

Sprint 394 showed that optimizing hash-router selection alone does not move
the broader HC-current/router boundary. The remaining measured route upload
cost is small but still repeated at every routed layer, and the current CPU
planner allocates pageable vectors and performs synchronous H2D route metadata
copies. This sprint tests whether a persistent pinned host workspace plus
stream-ordered async H2D uploads reduces that boundary without changing route
semantics.

## Scope

- Add `--route-plan-async-upload-gate` /
  `DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=1`.
- Keep the existing synchronous CPU planner as the default.
- Reuse the existing CPU route-plan semantics:
  - selected experts and weights remain produced by the model router.
  - per-rank offsets, route slots, route weights, and compact route plan match
    the existing compact-MoE layout.
  - the gate is limited to the compact-MoE model-router path.
- Use persistent pinned host buffers for selected experts, weights, offsets,
  route metadata, and compact plans.
- Upload route metadata with `cudaMemcpyAsync` on each destination rank stream
  so downstream kernels naturally wait in stream order.
- Wire the gate through the launcher and profile harness.
- Run a same-binary V100 HTTP A/B at the target `32` slot / `256K` real-router
  compact-MoE shape.

## Out Of Scope

- No PP/layer-split work.
- No GPU-side route planner promotion.
- No NCCL/collective rewrite in this sprint.
- No MTP work.
- No default promotion unless readiness, response parity, and performance
  gates pass.

## Definition Of Done

- The new gate is implemented and default-off.
- The launcher and profile harness can enable it.
- Emitted scaffold/status metadata records the gate state.
- The V100 `sm_70` binary builds on the pod.
- Same-binary HTTP A/B records readiness and response parity.
- Sprint docs record promote/reject with numbers.

## Risks

- Pinned async H2D may not help if route upload timing is dominated by CPU route
  construction or unavoidable router D2H.
- Stream-ordered upload is correctness-sensitive; host buffers must not be
  reused before copies complete.
- If route upload is already below noise, the gate should remain diagnostic
  and the next sprint should move to NCCL/collective work.

## Execution Plan

1. Implement persistent route-plan host workspace and async upload path.
2. Wire launcher/profile/matrix support.
3. Build on the V100 pod.
4. Run target-shape HTTP A/B with readiness and response parity.
5. Promote only if parity holds and server decode or GPU utilization improves.

## Outcome

Complete. Added and promoted `--route-plan-async-upload-gate` /
`DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=1`.

Implementation:

- Added a persistent pinned `RoutePlanHostWorkspace` for model-router route
  metadata.
- Copied router selected experts and route weights into pinned host buffers.
- Rebuilt the existing compact-MoE CPU route plan into persistent pinned
  buffers, preserving the previous route semantics.
- Uploaded offsets, route slots, route weights, and packed compact route plans
  with `cudaMemcpyAsync` on each destination rank stream.
- Guarded host-buffer reuse with per-rank upload completion events.
- Restored the control CUDA device after per-rank event creation; the first
  candidate attempt exposed this bug by allocating replicated control tensors
  on GPU 7 and failing VRAM startup.
- Wired the gate through:
  - `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  - `tools/ds4-v100-run-appliance.sh`
  - `tools/ds4-v100-tp-ep-profile.py`
  - `tools/ds4-v100-tp-ep-active-slot-matrix.py`

The launcher now defaults `DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=1`. The
profile harness defaults the gate on and exposes
`--disable-route-plan-async-upload` for future control runs.

## Validation

Local syntax/config:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py \
  tools/ds4-v100-tp-ep-active-slot-matrix.py
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Build passed with only existing unused-function warnings.

Same-binary V100 HTTP A/B:

Shape:

```text
32 requests / 32 slots / 256K ctx / position 262080 / 32 generated tokens
model-router routes / compact MoE / prompt-file soak / VRAM report
```

| Metric | Control | Route-plan async upload |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| Response parity | `32/32` | `32/32` |
| Candidate readiness | n/a | `true` |
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

Promote `route-plan-async-upload` as a TP/EP launcher/profile default. It
preserves response parity and readiness while improving the intended route
metadata boundary by about `30%` and improving server decode by about `2.15%`
in the target serving run.

This is a useful cleanup but not the final throughput lever. GPU utilization is
still only about `9.3%`, so the next sprint should return to NCCL/collective
work as requested and investigate whether the current peer-copy collective
transport is limiting TP/EP scheduling and graph/capture options.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint395-route-plan-async/http-control`
  - `/workspace/logs/sprint395-route-plan-async/http-candidate-route-plan-async`
  - `/workspace/logs/sprint395-route-plan-async/http-parity-summary.json`
  - `/workspace/logs/sprint395-route-plan-async/http-candidate-readiness.json`
- Local:
  - `logs/from-cluster/sprint395-route-plan-async`
