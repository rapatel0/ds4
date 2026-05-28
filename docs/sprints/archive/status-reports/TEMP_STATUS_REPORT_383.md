# TEMP Status Report 383

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 383 refreshed the real serving baseline at the target
`32` slot / `256K` context shape with GPU utilization and Sprint 382 VRAM
telemetry enabled.

## What Changed

- `tools/ds4-v100-tp-ep-active-slot-matrix.py` now records VRAM fields in the
  aggregate matrix:
  `vram_min_free_mib`, `vram_max_used_mib`, `vram_threshold_mib`, and
  `vram_failures`.
- The matrix runner has `--case-cooldown-seconds N` for repeated resident
  server startups.

## V100 Artifacts

```text
/workspace/logs/sprint383-vram-aware-matrix/
/workspace/logs/sprint383-vram-aware-matrix-retry/
/workspace/logs/sprint383-vram-aware-matrix-retry2/
/workspace/logs/sprint383-vram-aware-matrix-combined/
```

## Topline Matrix

`32` configured slots, `256K` context, `position=262080`, `32` generated
tokens/request, chat endpoint, GPU sampling at `500 ms`, VRAM reserve `64 MiB`.

| Active requests | HTTP 200 | Client tok/s | Server decode tok/s | Avg GPU util | Max GPU util | Max memory | Min free VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.321769 | 97.749438 | 8.540625% | 40.0% | 32398 MiB | 1754 MiB |
| 4 | 4/4 | 5.502740 | 97.092452 | 8.600694% | 39.0% | 32398 MiB | 1754 MiB |
| 8 | 8/8 | 10.891140 | 96.330654 | 8.375000% | 39.0% | 32398 MiB | 1754 MiB |
| 16 | 16/16 | 20.026059 | 92.690622 | 8.238372% | 39.0% | 32398 MiB | 1754 MiB |
| 32 | 32/32 | 43.853691 | 97.076706 | 9.292763% | 40.0% | 32398 MiB | 1754 MiB |

All completed cases had `vram_failures=0`.

## Startup Reliability Note

The no-cooldown matrix completed `requests=1`, then the next server startup
failed at:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:3187: out of memory
```

The first retry completed `requests=4`, then failed starting `requests=8` with
the same context-creation OOM. A 60-second inter-case cooldown completed
`8,16,32`. `nvidia-smi` showed `0 MiB` used after failure, so this looks like
CUDA context/teardown timing rather than true resident capacity exhaustion.

## Interpretation

The updated baseline confirms the prior thesis:

- Client aggregate tok/s scales with active requests because fixed decode cost
  is amortized.
- Server decode remains flat at roughly `92-98 tok/s`.
- Average GPU utilization remains below `10%`.
- GPU0 is still the busiest device; peers remain lightly utilized.
- Memory margin is tight by physical capacity but stable in allocator terms:
  `1754 MiB` minimum free at startup checkpoints.

## Next Best Work

Start an implementation sprint focused on steady-state launch/synchronization
overhead, not memory capacity or active-slot admission. The before/after gate
should use this Sprint 383 matrix as its baseline.
