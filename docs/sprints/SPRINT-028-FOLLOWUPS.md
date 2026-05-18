# SPRINT-028 Followups

## Critical

1. **Add an HTTP or process-serving wrapper around `ds4_v100_replay`**

   The replay runtime works, but the tool is one-shot. A usable appliance needs
   a long-running process that keeps the eight stages resident and accepts at
   least one deterministic request shape.

   - **Why**: Readiness still blocks on `public_serving`.
   - **Suggested sprint**: Sprint 029.
   - **Files**: new V100 server/tool wrapper, `ds4_v100_replay.*`,
     `tools/ds4-v100-gate.sh`.

2. **Add scheduler reset or explicit single-session semantics**

   `ds4_v100_replay` is intentionally one-shot because stage KV/cache state
   mutates during prompt replay. A server must either reset the state per
   request or expose one append-only session.

   - **Why**: Reopening stages per request costs minutes and defeats serving.
   - **Suggested sprint**: Sprint 029.
   - **Files**: `ds4_v100_scheduler.*`, `ds4_v100_replay.*`.

## High

3. **Reduce stage open/upload time**

   The first baseline spent about 281 seconds opening/uploading the eight
   stages. Parallel stage upload or a resident daemon is necessary before the
   appliance feels usable.

   - **Why**: Decode is now measurable, but startup dominates wall time.
   - **Suggested sprint**: Sprint 029 or Sprint 030.
   - **Files**: `ds4_v100_replay.c`, `ds4_v100_scheduler.c`, deployment
     scripts.

4. **Run longer decode baselines**

   The Sprint 028 gate proves counters and a short continuation. The next
   benchmark should run enough generated tokens to smooth output-head and
   handoff overhead.

   - **Why**: One continuation token is a baseline, not an optimization target.
   - **Suggested sprint**: Sprint 030.
   - **Files**: `tools/ds4-v100-replay.c`, benchmark scripts.

## Deferred

5. **MTP**

   Keep deferred until one-slot serving is resident and timed.

6. **Multi-slot scheduling**

   Keep deferred until the one-slot process is stable and reset/session
   semantics are explicit.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| HTTP/process serving wrapper | Critical | Sprint 029 | replay/server/gate |
| Scheduler reset or single-session semantics | Critical | Sprint 029 | scheduler/replay |
| Open/upload reduction | High | Sprint 029-030 | replay/scheduler/deploy |
| Longer decode baselines | High | Sprint 030 | replay/bench |
| MTP | Deferred | After resident serving | runtime/server |
| Multi-slot scheduling | Deferred | After single-slot serving | scheduler/server |
