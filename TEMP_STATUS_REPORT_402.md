# TEMP Status Report 402: NCCL VRAM Admission Guard

Date: 2026-05-26

## Current Focus

TP/EP only. No PP/layer-split work.

Sprint 402 finished the in-flight NCCL guardrail work. The goal was to stop
repeating expensive 32-slot/256K NCCL experiments that are functionally
correct at smaller shapes but run out of VRAM after communicator allocation.

## What Changed

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  - Added `--nccl-min-free-mib N`.
  - Added NCCL-gate detection for reduce-scatter compose, HC-current
    allgather, and attention-output allgather.
  - Added NCCL-specific VRAM checkpoints after rank/NCCL setup and after
    output-head allocation.
  - Emits `tp_ep_nccl_vram_admission_failed` and exits `14` when the NCCL
    reserve is violated.
- `tools/ds4-v100-run-appliance.sh`
  - Added `DS4_V100_TP_EP_NCCL_MIN_FREE_MIB`.
  - Defaults to `1536` only when an NCCL serving gate is active, otherwise `0`.
- `tools/ds4-v100-tp-ep-profile.py`
  - Added `--nccl-min-free-mib`.
  - Passes the guard through direct and launcher-backed profiling paths.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

All passed.

## V100 Results

| Case | Slots / ctx | NCCL gate | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Control | `32 / 256K` | off | `0` | `54639` | `93.773792` | `104.390813` | `1746 MiB` | Passed; NCCL guard ignored |
| HC-current NCCL | `32 / 256K` | on | `14` | n/a | n/a | n/a | `1114 MiB` | Correct early rejection |
| HC-current NCCL | `16 / 256K` | on | `0` | `54639` | `63.523008` | `72.165002` | `3820 MiB` | Passed; guard not overbroad |

The failing target-shape artifact now reports:

```text
tp_ep_nccl_vram_admission_failed label=nccl_after_output_head min_free_mib=1536
vram_nccl_after_output_head_min_free_mib=1114
vram_nccl_after_output_head_threshold_mib=1536
vram_nccl_after_output_head_failures=5
returncode=14
```

Artifacts:

- `logs/from-cluster/sprint402-nccl-vram-admission/direct-control/`
- `logs/from-cluster/sprint402-nccl-vram-admission/direct-nccl-fail/`
- `logs/from-cluster/sprint402-nccl-vram-admission/direct-nccl-pass-16/`

## Interpretation

This does not improve throughput. It turns the S400/S401 NCCL OOM pattern into
an explicit admission result. Narrow NCCL boundary substitutions remain
diagnostic-only; the next useful NCCL direction needs to be a broader
memory-planned TP/EP boundary that changes the topology rather than adding
isolated communicator overhead.
