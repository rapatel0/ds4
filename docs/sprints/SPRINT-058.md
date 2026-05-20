# Sprint 058: Replay Router Readback Suppression

## Status

Complete.

## Overview

Sprint 057 made token-step coalescing deterministic, but sustained decode still
measured about `3.66` generated tok/s at two slots with roughly `11%` average
GPU utilization. One remaining replay hot-path cost is CPU readback of
router-selected expert ids and weights in every FFN layer. That readback exists
for correctness reporting and validation, not for token generation: the
selected experts and weights are already device tensors consumed by the routed
MXFP4 kernels.

This sprint removes that synchronization point from the replay appliance path
while keeping direct scheduler and layer tests fully diagnostic by default.

## Goals

1. Add a runtime flag that controls router selected-expert/weight readback.
2. Default the replay appliance to skip readback on the generation hot path.
3. Preserve existing direct layer and scheduler diagnostic reports unless they
   explicitly opt in to suppression.
4. Verify selected-token correctness on the real 8x V100 model.
5. Re-run sustained decode at one and two slots and compare against Sprint 057.

## Out of Scope

- Persistent or tensor-core-friendly MoE kernel rewrites.
- Removing batched FFN input copy overhead.
- Changing router selection semantics.
- Enabling MTP draft commit.
- Public API changes beyond the internal replay/scheduler options.

## Implementation Notes

- `ds4_v100_replay_options` owns the appliance default.
- `ds4_v100_stage_scheduler_options` propagates the setting to every stage.
- `ds4_v100_layer_execute_config` controls the layer hot path.
- When suppression is enabled, `execute_ffn_delta` and
  `execute_ffn_delta_batch` keep selected experts and route weights on device.
  Reports still carry the route count, but selected ids and weights are zeroed.
- When suppression is disabled, the previous readback, selected-expert
  validation, and report population behavior is unchanged.

## Definition of Done

- `cc -fsyntax-only -I. ds4_v100_layer_execute.c` passes.
- `cc -fsyntax-only -I. ds4_v100_scheduler.c` passes.
- `cc -fsyntax-only -I. ds4_v100_replay.c` passes.
- `cc -fsyntax-only -I. tools/ds4-v100-replay.c` passes.
- Local object builds pass for touched C files.
- `git diff --check` passes.
- V100 `sm_70` build passes for `tools/ds4-v100-replay`.
- Real 8-GPU replay still selects first token hex `3136`.
- Sustained decode artifacts are captured under
  `logs/from-cluster/sprint058-router-readback-suppression`.
- Report compares Sprint 058 one-slot and two-slot throughput against Sprint
  057 default.
