# Sprint 458: TP/EP Graph Audit Telemetry and Locked HTTP Probe

## Objective

Make CUDA graph serving diagnostics visible in the permanent TP/EP HTTP A/B artifacts, then run a locked target-shape probe of persistent decode graph replay against the current promoted TP/EP baseline.

## Rationale

The next material throughput lever is launch/synchronization reduction in the TP/EP serving path. The runtime already emits `tp_ep_decode_cudagraph_audit`, but the profile and HTTP A/B summaries did not preserve those counters. That made graph attempts too hard to compare from the artifact JSON/Markdown.

Sprint 457 added the node-level lock; Sprint 458 uses it for the first graph probe so the result is not polluted by overlapping benchmarks.

## Scope

- Keep work strictly on the TP/EP path.
- Parse `tp_ep_decode_cudagraph_audit` into `summary.json`.
- Surface graph audit fields in `ab-summary.json` and `ab-summary.md`.
- Run the current 32-slot / 256K HTTP shape with the promoted rank-major/NCCL baseline on both legs and persistent decode graph only on the candidate.

## Definition of Done

- Local Python syntax checks pass.
- Remote Python syntax checks pass.
- The updated harness is synced to the V100 workspace.
- A locked HTTP A/B is attempted at the real serving shape.
- The result is recorded with an explicit promote/reject/blocker decision.
- `VISION.md` and `TEMP_STATUS_REPORT_458.md` are updated.

## Outcome

Implemented durable graph audit parsing in:

- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`

The first target-shape run exposed a profile-wrapper bug: the launcher inferred
the semantic post-attention path, but the profile wrapper forced
`DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=0`. The graph candidate then died
inside `collect_tensor_f32_stats` with:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9914:
operation not permitted when stream is capturing
```

Fixed the wrapper so semantic stats are skipped when enabled by the profile
default.

Final locked A/B artifact:

```text
/localpool/ds4/workspace/logs/s458-graph-audit-s32-t4-semstats
```

Shape:

```text
32 requests / 32 slots / 256K context / 4 generated tokens/request
```

| Metric | Control | Persistent graph candidate |
|---|---:|---:|
| readiness | pass | fail |
| response parity | - | 0/32 |
| HTTP 200 | 32 | 32 |
| server generated decode tok/s | 35.616755 | 54.056789 |
| server continuation decode tok/s | 35.499112 | 54.069704 |
| client generated tok/s | 5.198500 | 1.996102 |
| avg GPU util % | 12.067073 | 13.330952 |
| min free VRAM MiB | 1734 | 1200 |
| VRAM failures | 0 | 36 |
| output-head first token | 123477 | 32974 |
| graph capture attempted/succeeded | n/a | 43/43 |
| graph replay attempted/succeeded | n/a | 43/43 |
| graph replay ms | n/a | 587.940864 |
| graph blocker | n/a | none |

## Decision

Do not promote persistent graph serving yet.

This is still a useful result: target-shape HTTP capture/replay now works for
all 43 layers with zero graph stream-sync blockers, and it shows a real server
decode improvement. The blocker has moved to graph correctness and memory:
candidate responses are stale/different across decode positions, and graph
overhead drops the target-shape reserve below the `1536 MiB` NCCL threshold.

Next work should make decode-position and other dynamic graph inputs
device-resident/update-safe before replay, then re-test reserve after avoiding
unnecessary graph/capture residency.
