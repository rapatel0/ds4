# Sprint 369: TP/EP Durable Utilization Profiling

## Overview

Make the TP/EP profiling harness permanently capture GPU utilization and
memory evidence when requested, without adding runtime overhead to ordinary
serving or profile runs.

This sprint responds to the current operational concern: 32-slot TP/EP serving
is correct and stable, but observed GPU utilization remains low. Future kernel,
fusion, and TP/EP scheduling decisions need repeatable evidence attached to the
same profile artifact as throughput, compressed-KV timings, and coalescing
metadata.

## Scope

- Add an opt-in `nvidia-smi` sampling mode to
  `tools/ds4-v100-tp-ep-profile.py`.
- Write samples to `gpu_util.csv` inside each profile case directory.
- Add aggregate and per-GPU utilization/memory summaries to `summary.json`.
- Cover both HTTP serving profiles and direct token-major profiles.
- Keep the default disabled so normal profiling has no sampler process or
  polling overhead.

## Out Of Scope

- No PP/layer-split work.
- No MTP work.
- No kernel selection changes.
- No attempt to optimize utilization in this sprint; this sprint makes the
  measurement path durable.

## Implementation

- Add `--gpu-sample-interval-ms N` to the TP/EP profile script.
- When `N > 0`, start a background sampler during the measured HTTP request
  window or the direct subprocess window.
- Query:

```text
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

- Summarize:
  - `gpu_sample_count`
  - `gpu_util_avg`
  - `gpu_util_max`
  - `gpu_mem_used_max_mib`
  - `gpu_per_gpu`

## Definition Of Done

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py` passes.
- Shell launcher syntax check passes.
- Existing focused local tests still pass.
- V100 build of `tools/ds4-v100-tp-ep-full-layer-smoke` passes.
- A sampled TP/EP HTTP chat run writes `gpu_util.csv`.
- That run's `summary.json` contains utilization summary fields.
- Results are documented and committed.

## Notes

The sampler is intentionally coarse. Nsight/nvprof remain the kernel-level
tools; this addition makes every serving A/B carry enough utilization evidence
to tell whether a change improved real occupancy or only moved host/runtime
overhead around.
