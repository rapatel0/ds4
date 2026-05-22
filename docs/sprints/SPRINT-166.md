# Sprint 166 - Ready-Window Async Slot Coalescing

Date: 2026-05-21

## Objective

Test a middle ground between the two served scheduling modes that have already
been explored:

- default per-step async pipeline: preserves stage overlap but feeds one-slot
  routed kernels;
- fixed `DS4_V100_ASYNC_SLOT_CHUNK`: exposes dense routed kernels but makes
  each stage wait for an entire fixed chunk and regresses serving throughput.

This sprint adds an experimental ready-window coalescer to the per-step
pipeline. A stage waits for the current slot, then opportunistically batches
only contiguous slots that are already ready or become ready within a bounded
microsecond window. This should avoid the worst fixed-chunk stall while still
occasionally exposing multi-slot routed-FFN shapes.

## Scope

- Add opt-in env controls:
  - `DS4_V100_ASYNC_READY_CHUNK_MAX`
  - `DS4_V100_ASYNC_READY_WAIT_US`
  - `DS4_V100_ASYNC_READY_STAGE0_CHUNK_MAX`
- Keep `DS4_V100_ASYNC_SLOT_CHUNK` behavior unchanged when ready-window mode
  is unset.
- Do not change launcher defaults.
- Validate with scheduler smoke and one served 16-slot/256K A/B if the smoke
  passes.

## Implementation

1. Extend `replay_step_pipeline_worker_main()` in `ds4_v100_replay.c`.
2. When ready-window mode is off, preserve the existing fixed chunk behavior.
3. When ready-window mode is on:
   - stage 0 uses `DS4_V100_ASYNC_READY_STAGE0_CHUNK_MAX` or 1 by default;
   - stage N waits for the current slot from stage N-1;
   - stage N then collects contiguous ready slots up to
     `DS4_V100_ASYNC_READY_CHUNK_MAX`;
   - if `DS4_V100_ASYNC_READY_WAIT_US > 0`, stage N waits only until that
     deadline for the next contiguous slot before launching the current chunk.
4. Preserve existing event-handoff and synchronization behavior.
5. Record results in `docs/sprints/VISION.md` and cluster logs.

## Validation

Build:

```bash
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay tests/cuda_v100_stage_wavefront_smoke
```

Build result:

- PASS on the 8x V100 pod for `tools/ds4-v100-replay` and
  `tests/cuda_v100_stage_wavefront_smoke`.

Replay smoke:

- Baseline 4-slot/32K one-token replay passed:
  `generated_tokens_per_second=0.705770`, first token `48656c6c6f`.
- Ready-window 4-slot/32K one-token replay passed with:
  - `DS4_V100_ASYNC_READY_CHUNK_MAX=4`
  - `DS4_V100_ASYNC_READY_STAGE0_CHUNK_MAX=2`
  - `DS4_V100_ASYNC_READY_WAIT_US=200`
  - `generated_tokens_per_second=0.722567`, first token `48656c6c6f`.

Sustained served A/B, production TurboMind appliance path, MTP off:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Avg latency | Avg GPU util | Max GPU util |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-step event handoff control | 256K | 16 | 26.346605 | 24.699942 | 9203.042 ms | 31.514% | 66.000% |
| ready window, chunk 4, stage0 1, wait 0 us | 256K | 16 | 25.027658 | 23.463429 | 9683.286 ms | 30.058% | 60.000% |

The more aggressive `chunk 4, stage0 2, wait 200 us` variant also completed
all 16 requests correctly but was worse by response latency and utilization:
`avg_latency_ms=9580.359`, `max_latency_ms=11152.377`, `avg_gpu_util=26.315%`,
`max_gpu_util=49.000%`. The first timing harness run did not preserve exact
wall elapsed for that variant, so it is treated as diagnostic evidence rather
than the primary A/B row.

The sustained benchmark wrapper was also updated to accept `--appliance-dir`
so future runs can explicitly measure the production TurboMind appliance path
instead of accidentally omitting the TurboMind index. The current V100 pod does
not have `python3`, so the sustained A/B above used a small Perl socket client
while keeping the same resident replay server command.

## Definition of Done

- [x] Ready-window coalescing is implemented behind explicit env gates.
- [x] Existing fixed chunk behavior is unchanged when ready-window is unset.
- [x] V100 build passes.
- [x] Per-step smoke passes with ready-window unset.
- [x] Ready-window smoke passes with small chunk settings.
- [x] Served 16-slot/256K A/B is captured if smoke passes.
- [x] Result is recorded in the vision and cluster logs.
- [x] Changes are committed.

## Decision

Do not promote ready-window slot coalescing. It is correct and useful as an
explicit diagnostic, but both sustained candidates lost to the current
per-step event-handoff baseline at the practical 16-slot/256K tier. The next
meaningful throughput attempt should stop trying to recover dense routed
shapes through slot scheduling and move to a larger execution boundary:

- a persistent/fused routed-FFN kernel path that keeps packed MXFP4 dequant,
  gate/up HMMA, gated activation, down HMMA, and weighted reduce within one
  DS4-specific executor boundary; or
- a broader TP/EP scheduler design that makes denser routed-executor shapes
  without overlay-style per-layer copies.

## Risks

- The coalescing window may still reduce stage overlap enough to lose.
- Stage 0 chunking remains the critical tradeoff: chunking stage 0 creates
  denser early kernels but delays the first slot to stage 1.
- If this does not beat the baseline, the next scheduling work should stop
  trying to recover dense routed shapes through slot grouping and return to
  kernel-boundary work or a true TP/EP scheduler.
