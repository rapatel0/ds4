# Sprint 170 - Six-Route Routed Executor Probe

Date: 2026-05-21

## Objective

Test the routed-FFN executor shape used by the current best served path:

```text
ctx = 262144
slots = 16
async_pipeline_mode = per-step
per stage/layer call shape:
  total_routes = 6
  active_experts = 6
  max_routes_per_expert = 1
```

Sprints 158-169 showed that optimizing the 96/768/1536 route shapes does not
move the practical 256K appliance unless the scheduler also gives those shapes
to the hot path. The production path keeps stage overlap by processing one slot
at a time per stage, so this sprint adds a guarded fixed6 TurboMind gate/up
probe and measures whether dispatch bypass at the actual served shape helps.

## Scope

- Add `ggml_turbomind_ds4_mxfp4_gated_silu_6` to the TurboMind C ABI.
- Reuse the existing SM70 M16 MXFP4 probe kernel family.
- Add `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed6|chain6|ffn6|6`.
- Keep defaults unchanged.
- Select the fixed6 probe only when:
  - `total_routes == 6`
  - `num_experts == 6`
  - no token-index indirection is required
  - fused interleaved gate/up + gated-SiLU is active
  - `K == 4096`
- Build the TurboMind library and replay binary on the V100 pod.
- Run the production 16-slot/256K served A/B with per-step async/event
  handoff.

## Non-Goals

- No full fused gate/up/down persistent kernel yet.
- No MTP changes.
- No TP/EP topology changes.
- No default promotion unless the served A/B clearly improves.

## Definition of Done

- [ ] TurboMind shared library builds on the V100 pod.
- [ ] `tools/ds4-v100-replay` builds on the V100 pod.
- [ ] Launcher accepts `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed6`.
- [ ] Fixed6 selection is visible in a served or replay log.
- [ ] 16-slot/256K served A/B records generated and continuation tok/s.
- [ ] Result is recorded in `docs/sprints/VISION.md` and cluster logs.
- [ ] Changes are committed.

## Decision Gate

If fixed6 improves served continuation throughput outside run noise, keep it as
an opt-in candidate and use it to motivate a fuller six-route persistent
gate/up/down executor. If fixed6 is flat or slower, dispatch bypass is not the
missing lever; move directly to a true fused/persistent six-route executor or a
persistent TP/EP boundary.
