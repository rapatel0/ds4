# Sprint 169 - Explicit-Stream Routed-FFN Graph Capture

Date: 2026-05-21

## Objective

Resolve the specific Sprint 157 blocker: CUDA Graph replay around the
TurboMind routed-FFN boundary could not capture because the implementation used
the legacy default stream. Thread an explicit nonblocking stream through the
routed-FFN core and re-test whether graph launch replay provides useful V100
evidence without changing model math, packing, sharding, or defaults.

## Scope

- Keep the production default unchanged.
- Reuse the existing `DS4_V100_TURBOMIND_GRAPH=1` diagnostic gate.
- Add a per-GPU nonblocking graph stream and completion event for the
  TurboMind routed-FFN graph path.
- During graph capture, launch DS4 route-build, gather, compact-schedule,
  TurboMind gate/up, down, and scatter/reduce work on that stream.
- Launch captured graphs on the same stream and make the default stream wait
  before returning to the rest of the layer executor.
- Fall back to the existing path if stream setup or capture fails.

## Non-Goals

- No promotion to default in this sprint.
- No new MXFP4 math kernel.
- No whole-layer or whole-stage graph capture.
- No tensor-parallel topology rewrite.
- No MTP performance work.

## Implementation

1. Add per-device graph stream/event state in `ds4_cuda.cu`.
2. Add a small active-stream helper used only while graph capture is in
   progress.
3. Pass the active stream into the routed-FFN route builder, DS4 CUDA kernels,
   and TurboMind C ABI calls.
4. Capture and replay the existing graph on the explicit stream instead of
   stream 0.
5. Preserve normal default-stream behavior when graph replay is disabled.
6. Document build and V100 evidence.

## Definition of Done

- [x] `ds4_cuda.cu` builds locally for syntax.
      Local CUDA compilation is not available in this workstation shell, so
      the V100 pod build is the effective CUDA syntax gate.
- [x] `tools/ds4-v100-replay` builds on the V100 pod.
- [x] A graph-enabled production TurboMind smoke runs without legacy
      begin-capture failures.
- [x] The server or replay log reports either successful graph captures or a
      new explicit failure reason.
- [x] If smoke passes, a 128-slot/32K or 16-slot/256K A/B records generated and
      continuation/decode tok/s separately.
- [x] Result is recorded in `docs/sprints/VISION.md` and cluster logs.
- [x] Changes are committed.

## Implementation Notes

- Added per-GPU nonblocking graph streams plus start/done events for the
  TurboMind graph path.
- Threaded the active graph stream through route build, route gather/cast,
  compact-schedule construction, TurboMind gate/up/down calls, output clears,
  reductions, and scatter.
- CUDA Graph capture now runs on the graph stream instead of legacy stream 0.
- Graph launch now records a default-stream start event, makes the graph stream
  wait on it, launches the captured graph, records a graph-stream done event,
  and makes the default stream wait before returning.
- Graph capture is disabled when the experimental routed executor is enabled,
  because that path still performs active-expert host readback.

## V100 Evidence

Build:

```text
cd /workspace/ds4
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Short replay smoke, graph off:

```text
generated_tokens_per_second=1.907339
continuation_tokens_per_second=14.067746
tokens=19923,3,1730
```

Short replay smoke, graph on:

```text
generated_tokens_per_second=1.876273
continuation_tokens_per_second=17.747928
tokens=19923,3,1730
turbomind_graph captured entries present
begin_capture_failed=0
```

Direct replay 16-token A/B at 16-slot/256K:

| Mode | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---|
| graph off | `6.585787` | `16.039296` | token sequence matched |
| graph on | `6.058169` | `17.660156` | token sequence matched |

Served appliance 16-request A/B at 16-slot/256K, per-step async/event handoff:

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match | Graph log |
|---|---:|---:|---:|---:|---|
| graph off | `46.316374` | `43.421600` | `52.105920` | `16/16` | n/a |
| graph on | `40.058341` | `37.554695` | `45.065634` | `16/16` | `43` captures, `0` begin-capture failures, `0` launch failures |

Logs copied from the V100 pod:

```text
logs/from-cluster/sprint169-explicit-stream-graph/soak_graph_off/
logs/from-cluster/sprint169-explicit-stream-graph/soak_graph_on/
```

## Decision Gate

If explicit-stream capture succeeds and continuation/decode improves outside
run noise, keep it as an opt-in candidate for a broader routed-FFN execution
boundary. If capture still fails or replay is flat/slower, stop graph replay
work and move the next implementation to a true persistent routed-FFN executor
or a persistent TP/EP boundary.

## Decision

Explicit-stream graph capture fixes the Sprint 157 technical blocker, but it
does not improve practical served throughput. The short direct replay path
shows decode-only improvement once captures are resident, but the served
16-slot/256K appliance run regresses generated and continuation tok/s. Keep
`DS4_V100_TURBOMIND_GRAPH=1` as a diagnostic opt-in and do not promote it.

The next implementation should move to a larger execution boundary: either a
DS4-specific persistent routed-FFN executor that avoids per-layer host launch
and scheduling overhead, or a broader persistent TP/EP boundary that avoids the
copy-heavy per-layer TP overlay.
