# TEMP STATUS REPORT 415 — Spike B CUDA Graph Replay

Date: 2026-05-26

## Focus

Anchored back on `SPIKE_B_PLAN_ASSESSMENT.md`: TP/EP only, correct semantic path,
slots >= 4, CUDA graph/NCCL work before more eager-path tuning.

## Code Changes In Flight

- Added `--decode-cudagraph-replay-probe-gate`.
- The gate captures one decode step, instantiates the CUDA graph, launches it once,
  and uses graph output as the step result.
- Added replay counters and timings to per-layer output:
  `cudagraph_replay_attempted`, `cudagraph_replay_succeeded`,
  `cudagraph_instantiate_ms`, `cudagraph_replay_ms`.
- Kept default behavior off.
- Added stream-aware KV row-store path in `ds4_v100_tp_runtime.{cu,h}` so graph
  capture can keep KV writes on the active per-GPU streams.
- Converted the remaining capture-hostile peer copies on the semantic path to
  graph-safe kernels/event ordering in the TP/EP smoke runtime.

## Cluster Validation

Build:

- Built on `gpu-01` in the CUDA 12.2 devel container.
- Target: `sm_70`
- Command class: `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: build passed; only existing unused-kernel warnings.

Artifacts:

- Slot 4 replay: `/localpool/ds4/workspace/logs/spike-b-c-capture/replay-probe-alllayers-slot4-warm0/`
- Slot 8 replay: `/localpool/ds4/workspace/logs/spike-b-c-capture/replay-probe-alllayers-slot8-warm0/`

## Results

Both runs used the all-layer TP/EP serving harness with true DS4 attention output,
post-attention FFN input, NCCL HC-current allgather, NCCL attention-output
allgather, compressed KV path, and compact MoE decode.

| Slots | Capture | Replay | Replay sum | Projected slot tok/s | Graph nodes |
|---:|---:|---:|---:|---:|---:|
| 4 | 43/43 | 43/43 | 94.301 ms | 42.4 | 46,134 |
| 8 | 43/43 | 43/43 | 128.433 ms | 62.3 | 57,758 |

The wall time is much larger because this is still a probe that captures and
instantiates every layer in the process. The important number is `sum_replay_ms`,
which isolates graph launch execution after capture/instantiate.

## Interpretation

This proves the correct semantic all-layer TP/EP path is CUDA-graph launchable.
That is a material milestone: the previous state was capture eligibility only.

The first replay numbers are not yet production throughput. They still use one
graph capture/instantiate per layer per process invocation, no persistent graph
cache, no multi-token replay loop, and no output-head serving loop integration.

Slot scaling from 4 to 8 helps but is sublinear:

- 4 slots: 23.6 ms/layer aggregate average? No; full 43-layer replay is 94.3 ms.
- 8 slots: full 43-layer replay is 128.4 ms.
- Aggregate slot tok/s improves from 42.4 to 62.3.

This suggests graph replay removes a major launch barrier, but the path is still
dominated by the actual kernels/collectives and/or fixed per-layer work. The next
test should be persistent replay across multiple decode steps, then profiling the
heavy layers under replay.

## Multi-Step Probe

Artifact:

- `/localpool/ds4/workspace/logs/spike-b-c-capture/replay-repeat-slot8-steps8/`

I added a synthetic repeat-launch loop inside the replay probe, but the token-major
all-layer harness currently calls the per-layer decode loop with one decode step at
a time. So the attempted `--decode-steps 8` run still recaptured each layer each
outer token step instead of reusing graph execs.

Result:

- Steps 0-2 progressed with capture+replay.
- Step 3 failed at layer 2.
- Error:
  `store_f32_device_to_f8_kv_rows_kernel: operation would make the legacy stream depend on a capturing blocking stream`
- Failing stage:
  `tp_ep_compressed_kv_projection_failed layer 2 rc 16`

Interpretation:

- One-shot capture+replay is proven.
- Recapturing every token is the wrong architecture and exposes new stream-capture
  dependency hazards in the compressed-KV path.
- The next implementation has to cache `cudaGraphExec_t` per layer and replay it
  across token steps with persistent device buffers, rather than entering capture
  repeatedly in the token-major loop.

## Known Caveats

- Current probe uses `skip_decode_checksum=1` in the token-major serving scaffold,
  so this is launchability/performance evidence, not the final parity gate.
- The audit line previously reported `capture_eligible=0` because of a stale static
  helper-blocker heuristic even though capture+replay succeeded. I patched this
  locally so actual full capture/replay success makes the audit report eligible.
- The single-layer harness is not valid for the correct post-attention shared-FFN
  path because shared dense ops are initialized by the token-major all-layer path.

## Next Tasks

0. Apply the prior CUDA graph lessons from `TEMP_GRAPH_PRIOR_INSIGHTS.md`: no
   steady-state recapture, no pointer drift inside graphs, explicit graph/fallback
   counters, dynamic state through fixed device metadata buffers.
1. Rebuild after the audit-line cleanup and rerun the slot-4 replay probe once to
   confirm the report now says `capture_eligible=1`.
2. Add persistent graph exec storage per layer so capture/instantiate is setup cost,
   not per decode step.
3. Run a multi-step replay loop for slots 4 and 8 and report continuation tok/s.
4. Add parity gate against the existing peer-copy/oracle path for at least layer 2
   ratio-4 emit position.
5. Run Nsight/NCU on replay mode to identify the real post-graph bottleneck.
