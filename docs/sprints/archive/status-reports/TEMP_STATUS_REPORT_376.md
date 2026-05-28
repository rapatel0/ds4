# TEMP Status Report 376: Decode CUDA Graph Gate

Date: 2026-05-25

## Current Focus

Use `TEMP_THROUGHPUT_PROMPT.md` as the active performance plan. Sprint 376 is the S-A gate: `--decode-cudagraph-gate`.

The goal is not to fake a CUDA graph flag. The goal is to prove whether the 32-wide TP/EP decode step is capturable and whether graph replay can raise the current low-utilization serving path.

## What Changed

- Added `--decode-cudagraph-gate` to `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added launcher/profile plumbing:
  - `DS4_V100_TP_EP_DECODE_CUDAGRAPH=1`
  - `tools/ds4-v100-tp-ep-profile.py --decode-cudagraph`
- Added an initial aggregate graph audit line:
  - `tp_ep_decode_cudagraph_audit ...`
- Updated `docs/sprints/VISION.md` and `docs/sprints/STATUS.md` to make `TEMP_THROUGHPUT_PROMPT.md` the active performance steering source.

## V100 Build

Passed on gpu-01:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Initial V100 Audit

Command shape:

```text
32 slots / 256K context / position 262080 / 1 generated token / direct token-major
```

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/direct-audit/none-direct-decode-cudagraph
```

Topline result:

| Metric | Value |
|---|---:|
| Return code | `0` |
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

## Interpretation

CUDA graph replay is not ready yet. The blocker is inside the steady 43-layer decode step, not just output-head D2H. The current path has four broad `sync_all` waits per layer; each broad wait synchronizes eight rank streams and eight dense streams.

This validates the throughput prompt's warning that host-side synchronization must be removed before graph capture. It also narrows the next implementation task: replace these broad waits with CUDA event dependencies and rerun the same audit.

## Next Work

1. Replace in-step broad `sync_all` waits with stream/event dependencies where semantics allow it.
2. Extend the audit if needed to identify any remaining non-`sync_all` host waits in sub-stages.
3. Rebuild on the V100 pod.
4. Rerun the same direct audit.
5. Attempt graph capture only if the audit becomes clean enough to test honestly.

## Event-Barrier Audit Pass

Implemented a first gated stream-ordering replacement for the broad top-level `sync_all()` calls. Under `--decode-cudagraph-gate`, those waits now enqueue cross-GPU CUDA event dependencies instead of host `cudaStreamSynchronize` calls. This is still diagnostic-only and default-off.

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/event-barrier-audit/none-direct-decode-cudagraph
```

| Metric | Initial audit | Event-barrier audit |
|---|---:|---:|
| Generated decode tok/s | `82.384054` | `44.247981` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `sync_all_calls` | `172` | `0` |
| `event_barrier_calls` | n/a | `172` |
| `rank_stream_sync_count` | `1376` | `0` |
| `dense_stream_sync_count` | `1376` | `0` |
| `copy_stream_sync_count` | `0` | `0` |
| `helper_host_sync_blocker_classes` | n/a | `7` |
| `capture_eligible` | `0` | `0` |
| Blocker | `host_stream_synchronization` | `helper_host_synchronization` |

Interpretation: the ordering substitution preserved token/checksum parity and removed the audited top-level host stream waits, but it is not a performance win before graph replay. The remaining graph blockers are inside helper stages such as HC-current input, attention projection/state, compressed KV, typed history/raw read, and final HC expansion. The next useful work is to convert the hottest helper-level host waits to stream/event dependencies, starting with `run_shared_hc_current_input` and final HC expansion.

## HC-Current Event-Ordering Pass

Converted the main `run_shared_hc_current_input` host waits to stream/event ordering under `--decode-cudagraph-gate`. This keeps the default path unchanged.

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/hc-current-event-audit/none-direct-decode-cudagraph
```

| Metric | Event barrier | HC-current event pass |
|---|---:|---:|
| Generated decode tok/s | `44.247981` | `49.429146` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `sync_all_calls` | `0` | `0` |
| `event_barrier_calls` | `172` | `172` |
| `rank_stream_sync_count` | `0` | `0` |
| `dense_stream_sync_count` | `0` | `0` |
| `helper_host_sync_blocker_classes` | `7` | `6` |
| `capture_eligible` | `0` | `0` |
| Blocker | `helper_host_synchronization` | `helper_host_synchronization` |

HC-current timing improved materially inside the graph-gated path: `sum_pre_ep_hc_current_ms` moved from `47.654108` ms to `18.389539` ms and HC-current gather/fill subtimings dropped. Total decode is still slower than the initial host-sync control because the graph-gated path still pays broad event-barrier enqueue overhead and remains blocked by six helper classes.

Next helper target: final HC expansion and attention projection/state helpers. Capture should still not be attempted until helper blocker classes reach zero or the remaining blockers are explicitly accepted as non-capturable.

## Final-HC Event-Ordering Pass

Converted `run_shared_hc_final_expand` host waits to stream/event ordering under `--decode-cudagraph-gate`, including graph-gated control-stream launch and deferred host pointer swaps. The default path is unchanged.

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/final-hc-event-audit/none-direct-decode-cudagraph
```

| Metric | HC-current event pass | Final-HC event pass |
|---|---:|---:|
| Generated decode tok/s | `49.429146` | `48.189878` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `sync_all_calls` | `0` | `0` |
| `event_barrier_calls` | `172` | `172` |
| `rank_stream_sync_count` | `0` | `0` |
| `dense_stream_sync_count` | `0` | `0` |
| `helper_host_sync_blocker_classes` | `6` | `5` |
| `capture_eligible` | `0` | `0` |
| Blocker | `helper_host_synchronization` | `helper_host_synchronization` |

The final-HC stage itself improved (`sum_final_hc_ms` from `88.910098` to `74.314952` ms), while overall diagnostic decode was roughly flat/slightly lower because remaining broad event barriers and attention/helper waits still dominate. This validates the pointer-swap approach for this path but still does not make graph capture eligible.

Next helper target: attention projection and attention state/output helpers, then compressed-KV helper waits.

## Attention-Projection Event-Ordering Pass

Converted `run_true_ds4_attention_projection_prefix` host waits to stream/event ordering under `--decode-cudagraph-gate`, and skipped its diagnostic tensor-stat syncs while the graph gate is active. The default path is unchanged.

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/attention-projection-event-audit/none-direct-decode-cudagraph
```

| Metric | Final-HC event pass | Attention-projection event pass |
|---|---:|---:|
| Generated decode tok/s | `48.189878` | `45.864458` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `sync_all_calls` | `0` | `0` |
| `event_barrier_calls` | `172` | `172` |
| `rank_stream_sync_count` | `0` | `0` |
| `dense_stream_sync_count` | `0` | `0` |
| `helper_host_sync_blocker_classes` | `5` | `4` |
| `capture_eligible` | `0` | `0` |
| Blocker | `helper_host_synchronization` | `helper_host_synchronization` |

This removes one more graph blocker class and preserves token/checksum parity. The standalone graph-gated path remains slower before graph replay, mostly because the diagnostic event barriers are broad and attention/compressed helper waits remain.

Next helper target: attention state/raw-read/output helpers, then compressed-KV helper waits.

## Raw-Read Event-Ordering Pass

Removed host stream waits and layer<=2 tensor-stat syncs from `run_true_ds4_attention_raw_read` / `run_true_ds4_attention_raw_window` under `--decode-cudagraph-gate`. The default path is unchanged.

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/raw-read-event-audit/none-direct-decode-cudagraph
```

| Metric | Attention-projection event pass | Raw-read event pass |
|---|---:|---:|
| Generated decode tok/s | `45.864458` | `54.144225` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `sync_all_calls` | `0` | `0` |
| `event_barrier_calls` | `172` | `172` |
| `helper_host_sync_blocker_classes` | `4` | `3` |
| `capture_eligible` | `0` | `0` |
| Blocker | `helper_host_synchronization` | `helper_host_synchronization` |

The raw-read stage improved from `19.437900` ms to `4.487099` ms in the graph-gated diagnostic, and parity remained stable. Remaining helper blockers at that point were attention state, typed-history, and compressed-KV.

## Attention-State Event-Ordering Pass

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/attention-state-event-audit/none-direct-decode-cudagraph
```

| Metric | Raw-read pass | Attention-state pass |
|---|---:|---:|
| Generated decode tok/s | `54.144225` | `60.525660` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `helper_host_sync_blocker_classes` | `3` | `2` |
| `capture_eligible` | `0` | `0` |

## Typed-History Event-Ordering Pass

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/typed-history-event-audit/none-direct-decode-cudagraph
```

| Metric | Attention-state pass | Typed-history pass |
|---|---:|---:|
| Generated decode tok/s | `60.525660` | `59.120555` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `helper_host_sync_blocker_classes` | `2` | `1` |
| `capture_eligible` | `0` | `0` |

## Compressed-KV Event-Ordering Pass

Artifact path:

```text
logs/from-cluster/sprint376-decode-cudagraph/compressed-event-audit/none-direct-decode-cudagraph
```

| Metric | Typed-history pass | Compressed-KV pass |
|---|---:|---:|
| Generated decode tok/s | `59.120555` | `54.788890` |
| Output first token | `54639` | `54639` |
| Output checksum | `24071637347` | `24071637347` |
| Scaffold checksum | `3401922407` | `3401922407` |
| `helper_host_sync_blocker_classes` | `1` | `0` |
| `capture_eligible` | `0` | `1` |
| Blocker | `helper_host_synchronization` | `none` |

Current result: the graph-gated direct one-step non-emitted-row decode shape
is now audit-clean enough to attempt real CUDA graph capture. The next work is
not promotion; it is a capture attempt and the CUDA error/blocker report if
capture rejects any operation.

## Real Capture Attempts

After the audit reported `capture_eligible=1`, I attempted real stream capture
on the V100 pod at the same target shape:

```text
32 slots / 256K context / position 262080 / 1 generated token / direct token-major
```

Build still passes on gpu-01:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Capture attempt artifacts:

```text
logs/from-cluster/sprint376-decode-cudagraph/capture-attempt-audit/none-direct-decode-cudagraph
logs/from-cluster/sprint376-decode-cudagraph/capture-attempt-unified/none-direct-decode-cudagraph
logs/from-cluster/sprint376-decode-cudagraph/capture-attempt-seed/none-direct-decode-cudagraph
logs/from-cluster/sprint376-decode-cudagraph/capture-attempt-peer-kernel/none-direct-decode-cudagraph
```

Results:

| Attempt | Return | CUDA error | Interpretation |
|---|---:|---|---|
| separate captures for all streams | `2` | `operation would result in a merge of separate capture sequences` at line `4724` | Independent per-stream/per-rank captures cannot represent the existing cross-stream event waits. |
| single root stream capture | `2` | `dependency created on uncaptured work in another stream` at line `4724` | Root capture alone does not pull rank streams into the graph soon enough. |
| root capture with seed event joining all streams | `2` | `operation not permitted when stream is capturing` at line `4820` | The first `cudaMemcpyPeerAsync` in HC-current split broadcast is not graph-capturable on this stack. |
| HC-current peer copies replaced with graph-gated device copy kernels | `2` | `operation not permitted when stream is capturing` at line `8561` | Capture advances past HC-current copies, then fails at the next `cudaMemcpyPeerAsync` in attention projection. |

Decision: **REJECT graph replay as a Sprint 376 promotion path**. The blocker
is no longer generic host synchronization; it is that the current TP/EP decode
step uses many `cudaMemcpyPeerAsync` transfers, and this V100/CUDA stack rejects
those operations during stream capture. A full solution would require a broader
P2P transport rewrite to kernel/UVA copies or a different graph topology. That
is larger than Sprint 376 and not the next highest-confidence throughput step.

Next work per `TEMP_THROUGHPUT_PROMPT.md` and `docs/sprints/VISION.md`:
start the next isolated gate, `--batched-paged-attn-gate`, to reduce launch
count and fragmented attention/KV work without depending on CUDA graph capture.
