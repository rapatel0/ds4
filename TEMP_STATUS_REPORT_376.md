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
