# TEMP Status Report 401

Date: 2026-05-26

## Focus

Sprint 401 tested a default-off NCCL allgather path for the TP/EP HC-current
hidden-state boundary. This stayed strictly on the TP/EP codepath.

## What Changed

- Added `--tp-hc-current-input-nccl-allgather-gate`.
- Added launcher/profile wiring:
  `DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER`.
- Reused the existing NCCL communicator lifecycle when the gate is active.
- Added a rank-major NCCL receive buffer per rank.
- Added a rank-major-to-slot-major conversion kernel so NCCL output can feed
  existing dense input fill and route-pack logic.
- Fixed the device-context handoff after NCCL: GPU0 control-stream kernels now
  explicitly reset the current CUDA device to GPU0 before launching.

## V100 Results

Build passed:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Target `32` slot / `256K` direct A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | HC gather ms | HC fill/pack ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 54639 | 85.897762 | 99.733266 | 6.986851 | 28.665986 | 745.071798 | 1746 MiB |
| NCCL HC-current allgather | 2 | n/a | n/a | n/a | partial | partial | partial | 1114 MiB |

The candidate reached real layers after the device-context fix, then failed on
raw-SWA allocation:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9804: out of memory
```

The communicator memory delta is again about `+660 MiB/GPU`: rank-buffer max
used rose from `2317` to `2979` MiB, and post-output-head min free VRAM fell
from `1746` to `1114` MiB.

Functional `16` slot / `256K` direct A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | HC gather ms | HC fill/pack ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 54639 | 65.078267 | 73.863020 | 5.532507 | 17.896066 | 491.715616 | 4454 MiB |
| NCCL HC-current allgather | 0 | 54639 | 61.918746 | 69.068787 | 15.830067 | 8.442391 | 516.806332 | 3820 MiB |

The 16-slot run proves correctness but rejects performance. HC-current gather
regressed by about `2.86x`, and total decode regressed by about `5.1%`.

## Decision

Do not promote. Keep diagnostic-only.

This is now the second serving-facing NCCL gate that is correct but not useful
as a narrow boundary. NCCL still looks strong in proxy measurements for broad
TP hidden collectives, but production integration needs a shared communicator
memory plan and a larger fused TP/expert boundary.

## Artifacts

- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-control/`
- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-candidate/`
- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-control-16/`
- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-candidate-16/`

