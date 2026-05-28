# TEMP Status Report 458

## Current Focus

Persistent CUDA graph serving probe for the TP/EP appliance, with permanent graph audit telemetry in the A/B artifacts.

## Implementation

Updated:

- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`

Newly parsed graph audit fields include:

```text
graph_audit_sync_all_calls
graph_audit_stream_sync_count
graph_audit_capture_attempted
graph_audit_capture_succeeded
graph_audit_replay_attempted
graph_audit_replay_succeeded
graph_audit_sum_replay_ms
graph_audit_capture_eligible
graph_audit_blocker
```

## Validation

- Local `python3 -m py_compile`: pass
- Remote sync to `/localpool/ds4/workspace/ds4-sprint181`: pass
- Remote `python3 -m py_compile`: pass

## Cluster Probe

Artifact:

```text
/localpool/ds4/workspace/logs/s458-graph-audit-s32-t4-semstats
```

Shape:

```text
32 requests / 32 slots / 256K context / 4 generated tokens/request
```

Control:

```text
HC-current NCCL + routed FFN rank-major + model-router rank-major
```

Candidate:

```text
same baseline + decode cudagraph + persistent decode cudagraph
```

## Result

First attempt:

```text
/localpool/ds4/workspace/logs/s458-graph-audit-s32-t4
```

failed during graph capture because diagnostic semantic stats attempted a
`cudaStreamSynchronize` inside `collect_tensor_f32_stats`:

```text
operation not permitted when stream is capturing
```

Fixed by allowing the profile wrapper to enable semantic stats skipping for the
launcher-inferred semantic path.

Final locked target-shape result:

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

Decision: do not promote. Persistent graph serving is now operational enough to
capture and replay all 43 layers at the target shape, but it is not correct and
does not pass the 1536 MiB NCCL reserve.

Next focus: make graph dynamic state explicit and replay-safe. The candidate
looks like it is reusing stale captured state across decode positions, which
explains the repeated token pattern and parity failure despite graph replay
succeeding.
