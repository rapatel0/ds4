# Sprint 376: Decode CUDA Graph Capture Gate

## Overview

Implement the second throughput-pivot gate from
`TEMP_THROUGHPUT_PROMPT.md`: `--decode-cudagraph-gate`.

Sprint 371 showed that the TP/EP serving path is low-utilization and mostly
flat across active request counts at the target `32` slot / `256K` shape.
Sprint 375 proved that moving the output-head path to stream/event sequencing
preserves tokens but does not improve serving throughput. Sprint 376 is the
make-or-break test for the launch/synchronization thesis: can a shape-static
decode step be captured and replayed with CUDA graphs enough to raise GPU
utilization or server decode tok/s?

## Scope

- Add a default-off CLI gate:

```text
--decode-cudagraph-gate
```

- Add launcher/profile plumbing:

```text
DS4_V100_TP_EP_DECODE_CUDAGRAPH=1
tools/ds4-v100-tp-ep-profile.py --decode-cudagraph
```

- Audit graph capture eligibility for the token-major `run_one_step` path in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Capture and replay only the static per-rank decode-step body if eligibility
  is clean enough.
- Preserve current behavior when the gate is off.
- Validate on the V100 pod with same-binary A/B at the real serving shape.

## Out Of Scope

- Do not revive PP/layer-split work.
- Do not build a generic scheduler abstraction.
- Do not change model math, output-head selection, tokenization, or session
  semantics.
- Do not implement MTP in this sprint.
- Do not start paged attention, compact MoE, TP-sharded experts, or FP8 KV
  unless graph capture is proven infeasible and the sprint is explicitly
  closed with that result.

## Implementation Plan

### Phase 1: Capture Audit

Add a read-only graph audit mode under `--decode-cudagraph-gate` before
attempting capture:

- Count host syncs inside one token-major decode step.
- Count device allocations/frees inside the captured region.
- Identify host-read dependencies that affect kernel launch parameters.
- Identify kernels or API calls that are not graph-capturable.
- Emit a single audit line per run:

```text
tp_ep_decode_cudagraph_audit ...
```

The audit must distinguish:

- graph blockers inside `run_one_step`
- host waits outside `run_one_step`, especially output-head selected-token D2H
- diagnostic-only waits that are disabled in serving mode

### Phase 2: Persistent Graph Buffers

If the audit shows the step is capturable enough to proceed, add persistent
per-rank graph resources:

- graph input buffers
- graph output buffers
- captured per-rank stream/event dependencies
- graph handles and executable instances
- explicit teardown

All graph resources must be resident for the lifetime of the TP/EP server or
direct profile run. Replay must not depend on per-step host pointer mutation.

### Phase 3: Minimal Graph Replay

Capture the static 32-wide per-rank decode step behind the gate:

- one graph per rank, not one global host graph
- include stream waits/events needed by peer-copy compose where capturable
- keep output-head outside the graph for the first implementation
- require checksum-identical direct replay before HTTP testing

If capture fails, do not paper over it with a fake gate. Record the exact CUDA
error, call site, and remaining blocker in this sprint.

### Phase 4: V100 A/B

Build on the V100 pod:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run same-binary A/B:

- direct token-major smoke at `32` slots / `256K`
- HTTP active-slot matrix at `32` active requests / `32` slots / `256K` /
  `32` generated tokens/request
- GPU utilization sampling enabled

## Validation Fields

Every result must report:

- active requests
- configured slots
- context
- generated/decode tok/s
- continuation tok/s
- average and max GPU utilization
- first token
- all-layer decode checksum
- graph audit counts
- graph capture status
- graph replay count
- CUDA graph errors, if any

## Definition Of Done

- `--decode-cudagraph-gate` builds and defaults off.
- Launcher/profile plumbing exists and defaults off.
- V100 build passes.
- Capture audit emits remaining graph blockers.
- If capture succeeds, graph replay preserves first token and decode checksum.
- Same-binary V100 A/B artifacts are copied to:

```text
logs/from-cluster/sprint376-decode-cudagraph
```

- Sprint doc records an explicit PROMOTE, KEEP-OPT-IN, or REJECT decision.
- `TEMP_STATUS_REPORT_376.md` summarizes the result.
- `docs/sprints/VISION.md` and `docs/sprints/STATUS.md` are updated.
- Changes are committed.

## Decision Rule

Promote only if graph replay preserves first token/checksum and materially
improves GPU utilization or server decode tok/s at the real `32` slot /
`256K` serving shape.

Keep opt-in if graph replay works and preserves correctness but the performance
result is flat or noisy.

Reject if capture changes tokens/checksum, fails under normal serving shape, or
adds enough overhead to regress HTTP serving. If capture is blocked before
replay, close the sprint with the blocker list and use that list to choose the
next sprint rather than silently continuing into unrelated kernel work.

## Progress

### Initial Capture Audit

Implemented the default-off CLI/env/profile plumbing and a first
`tp_ep_decode_cudagraph_audit` line. V100 build passed:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct target-shape audit:

```text
tools/ds4-v100-tp-ep-profile.py \
  --run-mode direct-token-major \
  --tool none \
  --artifact-dir /workspace/logs/sprint376-decode-cudagraph/direct-audit \
  --tokens 1 \
  --position 262080 \
  --slots 32 \
  --decode-cudagraph
```

Result:

| Field | Value |
|---|---:|
| Return code | `0` |
| Slots/context | `32` / `256K` |
| Generated decode tok/s | `82.384054` |
| Output first token | `54639` |
| Output checksum | `24071637347` |
| Scaffold checksum | `3401922407` |
| `sync_all_calls` | `172` |
| `rank_stream_sync_count` | `1376` |
| `dense_stream_sync_count` | `1376` |
| `copy_stream_sync_count` | `0` |
| `capture_eligible` | `0` |
| Blocker | `host_stream_synchronization` |

Artifacts:

```text
logs/from-cluster/sprint376-decode-cudagraph/direct-audit/none-direct-decode-cudagraph
```

Interpretation: graph replay is not honestly attemptable yet. The counted
`sync_all` calls are in the steady token-major decode step, not only around
the output head. At the current target path this is `4` broad synchronization
points per layer for `43` layers, and each point waits rank streams plus dense
streams across all `8` GPUs.

Next implementation step: replace the in-step broad host synchronizations with
stream/event dependencies where semantics allow it, then rerun this audit. If
the remaining blockers move into non-capturable library calls or required
host-read dependencies, close Sprint 376 with that explicit blocker list and
move to the next throughput-prompt gate.
