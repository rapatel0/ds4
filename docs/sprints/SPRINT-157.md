# Sprint 157 - Routed-FFN CUDA Graph Replay

Date: 2026-05-21

## Objective

Start the next larger routed-FFN executor thread with an implementation that can
remove host launch/orchestration overhead without changing the TurboMind MXFP4
packed layout.

Sprint 156 ruled out host-orchestrated stream-per-expert pipelining as a
production win. This sprint adds an opt-in CUDA Graph replay path around the
post-router TurboMind routed-FFN core, then tests whether replaying the fixed
128-slot/32K and 256-slot/16K FFN shape moves served decode throughput.

## Scope

- Keep the default production path unchanged.
- Add `DS4_V100_TURBOMIND_GRAPH=1` as an explicit experimental flag.
- Capture one graph per GPU/layer/shape/pointer set after one warmup call.
- Force the graph capture call through the no-readback TurboMind total-token
  ABI so the captured segment does not contain the old synchronous offset
  readback path.
- Fall back to the normal implementation if capture fails, scratch moves, route
  validation/profile/group-pipeline is enabled, or the TurboMind ABI is missing.
- Benchmark prompt, generated, and continuation/decode tok/s separately.

## Non-Goals

- No default promotion without V100 served A/B.
- No new MXFP4 math kernel in this sprint.
- No graph capture of the whole layer executor, router host bookkeeping, token
  upload, or shared FFN path yet.
- No broad tensor-parallel scheduler work.

## Definition Of Done

- Build passes on the V100 pod.
- Full 43-layer scheduler smoke passes with graph flag off.
- Full 43-layer scheduler smoke passes with `DS4_V100_TURBOMIND_GRAPH=1`.
- Served A/B is run at 128-slot/32K and 256-slot/16K if correctness passes.
- Results are recorded in `TEMP_CURRENT_REPORT.md`,
  `TEMP_STATUS_REPORT.md`, `docs/sprints/STATUS.md`, and
  `docs/sprints/EXPERIMENT-STATUS.md`.

## Decision Gate

Promote nothing automatically. If CUDA Graph replay is correct and improves
continuation/decode throughput outside run noise, keep it as an opt-in
candidate for broader graph boundaries. If it is flat or capture-incompatible,
record that host launch replay is not enough and move to a true persistent
routed-FFN kernel or the bounded one-layer 2-way TP prototype.

## Initial Implementation

Implemented an opt-in graph cache in `ds4_cuda.cu`:

- keyed by GPU, fixed routed shape, gate/up/down offsets, selected/weight/input
  output pointers, graph-relevant flags, and accumulation mode;
- first call warms the existing path and populates scratch/table caches;
- second call captures and instantiates the routed TurboMind FFN core;
- later calls replay the graph if the global temp scratch pointer is unchanged;
- graph capture is skipped when profiling, route validation, or group-pipeline
  diagnostics are enabled.

Launcher/config additions:

- `DS4_V100_TURBOMIND_GRAPH=0`
- `DS4_V100_TURBOMIND_GRAPH_VERBOSE=0`

## Validation Log

Build/smoke:

- V100 build passed after the graph implementation.
- V100 build passed again after the single-slot scratch-address stabilization
  change.
- Full 43-layer scheduler smoke passed with graph disabled:
  `stages=8 ... slots=128 layers=43 tm_layers=43 ... ok`.
- Full 43-layer scheduler smoke passed with
  `DS4_V100_TURBOMIND_GRAPH=1`; because the smoke calls each layer shape once,
  it only emitted warmup keys and did not attempt replay.

Served 128-slot/32K:

| Run | Generated tok/s | Continuation/decode tok/s | Correctness | Captures |
|---|---:|---:|---|---:|
| Graph-disabled control | `59.607704` | `55.882222` | 128/128 | n/a |
| Graph + stable scratch, global capture | `59.450666` | `55.734999` | 128/128 | 0 |
| Graph + stable scratch, thread-local capture | `59.367233` | `55.656781` | 128/128 | 0 |

Capture diagnostics:

- Served logs emitted 43 `turbomind_graph warmup` keys for the 43 routed
  layers.
- Global capture then emitted 43 `begin capture failed` messages.
- Thread-local capture also emitted 43 `begin capture failed` messages.
- A small diagnostic with `--async-pipeline-mode off` remained correct but
  still emitted begin-capture failures, including both `tokens=1` and
  `tokens=16` graph keys.

Representative CUDA error:

```text
operation not permitted when stream is capturing
```

## Decision

Do not promote CUDA Graph replay. The wrapper-level graph probe is correct and
safe as an opt-in fallback path, but it does not capture in served mode and
therefore cannot improve throughput. The failure persists with the async
pipeline disabled, which points at the legacy default-stream launch structure
rather than only the serving scheduler.

If we want graph replay later, the next version needs an explicit stream
threaded through the TurboMind routed-FFN kernels and related CUDA operations.
That is a larger executor rewrite, not a small cache wrapper.

For the main optimization path, move on to a real persistent/larger fused
routed-FFN executor or the bounded 2-way TP prototype for the 128-slot/32K NV2
case.
