# TEMP Status Report 403: NCCL With Quantized KV Matrix

Date: 2026-05-26

## Current Focus

TP/EP only. No PP/layer-split work.

Sprint 403 tested whether the existing FP8 E5M2 KV gate could buy back enough
VRAM to make HC-current NCCL viable at the production target shape:
`32` slots / `256K` context.

## What Changed

Added a reusable matrix harness:

- `tools/ds4-v100-tp-ep-nccl-kv-matrix.py`

The harness runs direct-token-major profile cases and writes:

- per-case command/stdout/stderr artifacts,
- profile harness artifacts,
- `matrix-summary.json`,
- `matrix-summary.md`.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-nccl-kv-matrix.py
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-nccl-kv-matrix.py
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

All passed.

## V100 Matrix

Shape:

```text
ctx=262144
slots=32
position=262080
requests=32
max_requests=80
tokens=2
nccl_min_free_mib=1536
```

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | NCCL threshold | NCCL failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | `0` | `54639` | `98.076858` | `107.106917` | `1746 MiB` | n/a | `0` |
| FP8 E5M2 KV | `0` | `54639` | `93.927351` | `103.344304` | `1746 MiB` | n/a | `0` |
| HC-current NCCL | `14` | n/a | n/a | n/a | `1114 MiB` | `1536 MiB` | `5` |
| FP8 E5M2 KV + HC-current NCCL | `14` | n/a | n/a | n/a | `1114 MiB` | `1536 MiB` | `5` |

Artifacts:

- `logs/from-cluster/sprint403-nccl-kv-matrix/matrix-summary.json`
- `logs/from-cluster/sprint403-nccl-kv-matrix/matrix-summary.md`

## Interpretation

`--fp8-e5m2-kv` is not a memory-reclamation switch in this TP/EP path. The
runtime already defaults to FP8 E4M3 block-128 KV; the gate changes the FP8
flavor to E5M2. That is why FP8 E5M2 had identical minimum free VRAM to the
control and why FP8 E5M2 + NCCL failed at the same `1114 MiB` post-output-head
checkpoint as NCCL alone.

## Decision

Do not promote E5M2 KV or narrow HC-current NCCL. Keep both diagnostic-only.

The next NCCL work should either remove/reschedule at least about
`422 MiB/GPU` before the post-output-head NCCL checkpoint, or shift to a
broader NCCL TP boundary where communicator memory is offset by replacing a
material amount of peer-copy transport.
