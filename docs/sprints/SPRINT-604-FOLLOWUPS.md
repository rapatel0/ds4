# Sprint 604 Follow-Ups

## 1. Promote edges+fix as the launcher sync default (605 lead)

- **What**: With DENSE_FIX promoted, the s603 edges mode passes its
  correctness gate (edges+fix 2/2 clean in Phase E composition) and is
  faster than join+fix (~177 vs ~189 ms, S=8). s603 left edges unpromoted
  only on the +15% perf gate, but its real blocker was the correctness
  objection, now removed. Run a clean ≥30-run edges+fix soak (un-amplified,
  telemetry) and, if zero-token, flip `DS4_V100_TP_EP_S602_SYNC=edges`.
- **Severity**: Important (bankable ~+6-8% on a correct base).
- **Suggested sprint**: 605 first task.
- **Files**: launcher default; `engine/runtime_pack.cu` (edges already built).

## 2. Step-floor reduction campaign (the ≥50/slot path)

- **What**: The fix made the floor correct, not lower (~177 ms, S=8 ≈ 5.6
  tok/s/slot). Reaching the ~40-60 ms MTP-reachable floor needs the
  demonstrated levers stacked on the clean base: prefix launch compaction
  (~1.1 ms/layer — the attention/HC-current prefix, s601 follow-up), route-
  plan shadowing (~0.45, s599/603 C-C, never attempted), cross-layer graph
  consolidation. Each census-gated (the hazard is fixed, but new fast paths
  must still prove token-clean — the amplifier is now the cheap gate).
- **Severity**: Critical (the gating lever for the program target).
- **Suggested sprint**: 605-607.
- **Files**: `engine/hc_current.cu`, `engine/attention_output.cu`,
  `engine/router_plan.cu`, `engine/decode_loop.cu`.

## 3. Late-step (pre_compose) hazard class — confirm DENSE_FIX covers it

- **What**: The weaker late-step token class localized to the pre_compose
  dense↔rank site (same family). Phase D fix-on soak was fully clean, which
  suggests the fix's cross-GPU edge covers it, but it was confirmed via the
  attn_out_a amplifier, not a pre_compose-amplified fix-on run. Add one
  pre_compose-amplified fix-on gate to close it explicitly.
- **Severity**: Important (correctness completeness).
- **Suggested sprint**: 605 (cheap, the amplifier + flag exist).
- **Files**: `engine/runtime_pack.cu`, `engine/decode_loop.cu`.

## 4. Pod longevity / degradation

- **What**: The pod degraded over ~38h uptime (a run-stall truncated the
  soak from 52 to 34 pairs; day-to-day event-rate swings of 3-5x). Consider
  a periodic pod recycle between long sprints (the /workspace hostPath
  persists, so recreation is cheap — proven in s601), and keep per-run
  telemetry as standard.
- **Severity**: Important (measurement validity).
- **Suggested sprint**: operational, ongoing.

## 5. Amplifier is now a permanent correctness gate

- **What**: `DENSE_HAZARD_AMP` converts the statistical race gate into a
  1-run deterministic test. Every future fast-path change should be gated
  with it (amp-on must stay token-clean) rather than relying on rare
  natural events + large soaks. Keep it as the standard pre-promotion check.
- **Severity**: Nice-to-have (process improvement).
- **Files**: `engine/runtime_options.cuh`, `engine/runtime_pack.cu`.

## 6. Carry-forward from s602/s603

- Binary-default alignment (s602 #4): DENSE_FIX now defaults on in the
  binary too (done this sprint); the transport flags (relay/batched/kernel)
  still default off in the binary / on in the launcher — flip after a soak.
- NVIDIA escalation (s600/s602): the captured-NCCL race package stands;
  s604 is unrelated (engine ordering, now fixed).
- NO_SYS_RING doc correction (s602 #3): still pending.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| Promote edges+fix sync default | Important | 605 | launcher, runtime_pack.cu |
| Step-floor reduction campaign | Critical | 605-607 | hc_current/attention_output/router_plan/decode_loop |
| Confirm pre_compose class covered | Important | 605 | runtime_pack.cu, decode_loop.cu |
| Pod recycle between sprints | Important | operational | deploy/v100 manifest |
| Amplifier as standard gate | Nice-to-have | ongoing | runtime_options/runtime_pack |
| Binary-default + doc carry-forwards | Nice-to-have | 605/606 | runtime_options.cuh, env example |
