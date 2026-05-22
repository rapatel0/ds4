# TEMP Status Report 005

Date: 2026-05-22

## Current Topline

The current production appliance is functional on the V100 cluster, but not yet
near the practical throughput target.

| Run | Context | Slots | Tokens/request | Generated tok/s | Continuation tok/s | Match | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Sprint 181 production baseline | 256K | 16 | 64 | `48.163685` | `47.411127` | 16/16 | Best current sustained result |
| Sprint 181 single-slot baseline | 256K | 1 | 64 | `10.357728` | `10.195888` | 1/1 | Current single-slot production-pack result |
| Sprint 181 MTP verify | 256K | 16 | 2 | `7.013779` | `3.506890` | 16/16, MTP 16/16 | Correctness only; verify recomputes target |
| Sprint 182 async profile | 256K | 16 | 8 | `20.811726` | `18.210261` | 16/16 | Profiling enabled, not a speed run |
| Sprint 182 sync profile | 256K | 16 | 2 | `2.398485` | `1.199242` | 16/16 | Synchronized profile, diagnostic only |

The best current practical number remains about `47.4` continuation tok/s at
16 slots / 256K context. Profiling overhead and synchronized profile mode reduce
throughput substantially, so the Sprint 182 profile runs should not replace the
Sprint 181 baseline as the production topline.

## Current Cluster State

- Branch: `claude-takeover`
- Latest committed sprint: `b2ed5e7 sprint 181 persistent production pack`
- Active sprint: Sprint 182, production-pack profiling and profiler harness
- Build pod: `llm/llamacpp-build-8gpu`
- Node: `gpu-01`
- Pod IP: `10.1.55.90`
- Pod state: Ready
- Current GPU state: no active `ds4-v100-replay` process observed
- `/workspace` is backed by `/localpool/ds4/workspace`, not the host mirror
  drive
- Persistent optimized pack:
  `/workspace/packs/ds4-appliance-full-tm-gated-s181`
- Pack size: about `143G`

## What Was Tested Since The Last Report

Sprint 181 restored and validated the persistent production pack:

- Built TurboMind for `sm_70`.
- Built `tools/ds4-v100-appliance-pack`.
- Built `tools/ds4-v100-replay`.
- Regenerated the optimized gated-SiLU appliance pack on localpool storage.
- Verified 256K context, 16-slot serving correctness.
- Verified MTP sidecar compatibility in `verify` mode.

Sprint 182 added harness support for cleaner profiling:

- `tools/ds4-v100-sustained-decode-bench.sh` now supports
  `DS4_V100_REPLAY_BIN`.
- The same script now passes `--cuda-profiler-window` through to the replay
  server.
- `tools/ds4-v100-replay-nvprof-wrapper.sh` now uses
  `DS4_V100_REPLAY_UNDERLYING_BIN` for the real replay binary, avoiding
  recursion when the benchmark launches the wrapper via `DS4_V100_REPLAY_BIN`.
- Shell validation passed before syncing to the pod.

Sprint 182 completed two production-pack profile runs:

- Async per-step serving profile:
  `/workspace/logs/sprint182-production-pack-profile/profile-decode-256k-16slot/`
- Synchronized stage profile:
  `/workspace/logs/sprint182-production-pack-profile/profile-sync-256k-16slot/`
- nvprof wrapper smoke:
  `/workspace/logs/sprint182-production-pack-profile/nvprof-256k-smoke/`

## Sprint 182 Profile Findings

The async per-step profile is closest to production serving behavior:

- Context: `262144`
- Slots: `16`
- Tokens/request: `8`
- Requests: `16`
- Generated tok/s: `20.811726`
- Continuation tok/s: `18.210261`
- Prompt tok/s: `46.826384`
- Token match: `16/16`
- Average GPU util: `27.576%`
- Max GPU util: `63%`
- Average async total: `5831.286 ms`
- Average async host wait: `5809.689 ms`
- Handoff sum: `261.636 ms`

The synchronized profile is not representative for throughput, but it gives
useful stage visibility:

| Stage | Sum across stages |
|---|---:|
| Attention | `9315.361 ms` |
| FFN | `2223.435 ms` |
| HC attention transform | `630.391 ms` |
| HC FFN transform | `787.979 ms` |
| HC final transform | `47.113 ms` |
| Total profiled stage time | `13004.282 ms` |

At 256K context, attention/KV work is the largest visible synchronized-stage
cost. FFN still matters, but the profile weakens the idea that a routed-FFN
wrapper alone can deliver the major throughput jump.

## Interpretation

The appliance is executing correctly, resident on the V100 node, and using the
optimized production pack. The blocker is now execution architecture:

- GPU utilization remains too low for the target throughput range.
- The async pipeline preserves overlap but still exposes large host-wait and
stage-wait costs.
- The synchronized profile says long-context attention is a major part of the
current 256K decode cost.
- Prior routed-FFN wrapper experiments were flat, including the fixed6 shape
that matches the async per-step route pattern.
- MTP is still correctness/compatibility only because verify mode recomputes
the base target path.

## Current Decision Pressure

The next material implementation should probably be one of:

1. A true persistent fused routed-FFN boundary, not another launcher wrapper.
2. A persistent attention/KV boundary for the 256K serving shape.
3. A broader TP/EP topology that changes the execution shape enough to improve
   tensor-core occupancy and reduce per-stage serialization.

Given the Sprint 182 profile, option 2 or option 3 may deserve equal priority
with option 1. The profile shows that at the required context length, attention
is too large to treat as secondary.

## Remaining Before Practical Use

- Commit Sprint 182 documentation and copied profile evidence.
- Decide whether the next engineering sprint targets persistent FFN,
  persistent attention/KV, or TP/EP topology.
- Re-test best production throughput after the next implementation against:
  - 1 slot / 256K
  - 16 slots / 256K
  - a longer continuous serving run
  - MTP verify or commit only as a separate measurement
