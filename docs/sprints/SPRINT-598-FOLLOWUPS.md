# Sprint 598 Follow-Ups

## 1. Foreign GPU process appeared on gpu-01 during the sprint

- **What**: A transient host-level process (not a k8s pod) briefly held
  8.1 GiB on all 8 V100s mid-sprint, crashing one smoke run's VRAM reserve
  check. It self-cleared and never overlapped a measured window (verified
  per-run), but something outside Kubernetes can schedule onto the reserved
  node. Identify the source (host cron? another operator?) and decide
  whether the A/B lock needs a node-level guard.
- **Why**: Incident #2 during execution; a wait-for-idle preflight was added
  to the bench wrapper as mitigation.
- **Severity**: Important (silent bench-validity threat).
- **Suggested sprint**: operational — investigate on gpu-01 outside sprint
  scope.
- **Files**: bench wrapper preflight in `/workspace/s598-artifacts/COMMANDS.md`.

## 2. NCCL probe teardown hang (`ncclCommDestroy` with live graph exec)

- **What**: `tools/s598-nccl-capture-probe.cu` hangs at exit if comms are
  destroyed while the captured graph exec is alive; destroy order matters.
  The serving path is unaffected (comms outlive the graph), but document the
  ordering constraint in the probe and runtime teardown.
- **Severity**: Nice-to-have.
- **Suggested sprint**: on demand.
- **Files**: `tools/s598-nccl-capture-probe.cu`,
  `engine/runtime_resources.cu` (teardown order).

## 3. Binary default (`copy`) vs launcher default (`nccl`) divergence

- **What**: Promotion flipped the launcher env default; the Options struct
  default remains `copy`, so non-launcher invocations (smokes, probes,
  direct binary runs) silently run the slow rollback path. Either flip the
  binary default after a soak period or make non-launcher tools print the
  active transport.
- **Severity**: Nice-to-have (footgun for future benchmarking).
- **Suggested sprint**: 599 or 600 (after soak).
- **Files**: `engine/runtime_options.cuh`,
  `tools/ds4-v100-run-tp-ep-appliance.sh`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| Foreign GPU process on gpu-01 | Important | operational | (node-level) |
| Probe/teardown destroy-order hang | Nice-to-have | on demand | tools/s598-nccl-capture-probe.cu |
| Binary vs launcher transport default | Nice-to-have | 599/600 | engine/runtime_options.cuh |
