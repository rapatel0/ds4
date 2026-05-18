# SPRINT-029 Followups

## Critical

1. **Implement and validate MTP**

   The full gate now reports only `missing=mtp`. The next correctness milestone
   is adding the DS4 MTP draft/verify path to the appliance runtime without
   regressing the selected-token baseline.

   - **Why**: MTP is the last readiness blocker in the gate.
   - **Suggested sprint**: Sprint 030.
   - **Files**: `ds4_v100_replay.*`, scheduler/runtime files, MTP model loader
     or bridge files, `tools/ds4-v100-gate.sh`.

## High

2. **Parallelize resident stage open/upload**

   The HTTP process keeps weights resident after startup, but startup still
   takes about 5 minutes because the eight stage schedulers open/upload
   sequentially.

   - **Why**: Deployment and restart time dominate the user experience.
   - **Suggested sprint**: Sprint 030 or Sprint 031.
   - **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`.

3. **Run longer resident decode baselines**

   Sprint 029 validates one-token HTTP and two-token CLI replay. We need longer
   resident-generation runs to smooth output-head and handoff overhead.

   - **Why**: Short prompts are correctness evidence, not a meaningful
     optimization target.
   - **Suggested sprint**: Sprint 030.
   - **Files**: `tools/ds4-v100-replay.c`, benchmark scripts.

4. **Harden the serving API after MTP direction is clear**

   The current HTTP endpoint is intentionally narrow and sequential. Production
   use needs request limits, clear errors, process supervision, and possibly an
   OpenAI-compatible adapter.

   - **Why**: Public serving readiness is satisfied only for a loopback
     selected-token appliance smoke.
   - **Suggested sprint**: After MTP correctness or in parallel with deployment
     packaging.
   - **Files**: `tools/ds4-v100-replay.c`, deployment manifests/scripts.

## Deferred

5. **Multi-slot scheduling**

   Keep deferred until one-slot MTP and resident decode baselines are stable.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| MTP implementation/validation | Critical | Sprint 030 | replay/scheduler/gate |
| Parallel open/upload | High | Sprint 030-031 | replay/scheduler |
| Longer resident decode baselines | High | Sprint 030 | replay/bench |
| Serving API hardening | High | After MTP direction | replay/deploy |
| Multi-slot scheduling | Deferred | After one-slot MTP | scheduler/server |
