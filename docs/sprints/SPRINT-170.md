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

- [x] TurboMind shared library builds on the V100 pod.
- [x] `tools/ds4-v100-replay` builds on the V100 pod.
- [x] Launcher accepts `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed6`.
- [x] Fixed6 selection is visible in a served or replay log.
- [x] 16-slot/256K served A/B records generated and continuation tok/s.
- [x] Result is recorded in `docs/sprints/VISION.md` and cluster logs.
- [x] Changes are committed.

## Decision Gate

If fixed6 improves served continuation throughput outside run noise, keep it as
an opt-in candidate and use it to motivate a fuller six-route persistent
gate/up/down executor. If fixed6 is flat or slower, dispatch bypass is not the
missing lever; move directly to a true fused/persistent six-route executor or a
persistent TP/EP boundary.

## Outcome

Implemented and validated on the V100 pod (`llm/llamacpp-build-8gpu`, gpu-01).

- Added `ggml_turbomind_ds4_mxfp4_gated_silu_6` to the TurboMind C ABI
  (`api.cc`, `ggml-turbomind-ds4-probe.cu`, `include/ggml-turbomind-api.h`),
  reusing the SM70 M16 MXFP4 probe kernel.
- Added `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed6` in `ds4_cuda.cu` under the
  required guards (`total_routes==6`, `num_experts==6`, no token indirection,
  fused interleaved gate/up gated-SiLU, `K==4096`, `fused_n==4096`).
- Updated the `tools/ds4-v100-run-appliance.sh` launcher allowlist to accept
  `fixed6`. This was not called out in Scope but was required by the DoD; the
  A/B failed to start until it was added.
- Rebuilt the TurboMind `.so` and `tools/ds4-v100-replay` (`sm_70`); the new
  symbol is exported (`nm -D ... -> T ggml_turbomind_ds4_mxfp4_gated_silu_6`).

Same-binary served A/B at 16-slot/256K, per-step async + event handoff:

| Mode | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|
| control (executor off) | `44.454879` | `41.676449` | `16/16` |
| fixed6 | `44.344945` | `41.573386` | `16/16` |

`fixed6` was selected and executed on the real served 6-route shape (server log:
`fixed6 shape total_routes=6 active_experts=6 max_routes_per_expert=1` and
`selected fixed gate_up total_routes=6`), unlike the Sprint 158 `fixed96` probe
which never matched the served HTTP shape. Evidence:
`logs/from-cluster/sprint170-fixed6-routed-executor/`.

## Decision

`fixed6` is correct but throughput is flat-to-slightly-slower versus same-binary
control, so per the decision gate, dispatch bypass is **not** the missing lever
-- now established at the actual served shape, not just the scheduler-coalesced
96-route shape. Keep `DS4_V100_TURBOMIND_ROUTED_EXECUTOR` explicit opt-in
(default `off`); do not promote `fixed6`.

The next implementation moves to a larger execution boundary: a true
persistent/fused six-route routed-FFN executor (gate/up + activation + down +
weighted reduce as one boundary) or a persistent TP/EP boundary. See
`docs/sprints/SPRINT-170-FOLLOWUPS.md`.
