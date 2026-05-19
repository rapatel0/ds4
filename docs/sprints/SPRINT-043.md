# Sprint 043: Production Deployment Package

## Status

Complete.

## Overview

Sprint 043 closes the current `production_deployment` readiness blocker by
turning the verified one-slot V100 appliance into an operator-owned service
package. The package should be runnable on the 8x V100 host through an explicit
environment file, launcher, supervisor manifest, health/status/metrics probes,
and a deployment smoke that starts the same resident HTTP service operators will
use.

This sprint does not change model math. It does not claim MTP speculative
serving or throughput optimization. The served production mode remains the
verified base one-slot appliance while the MTP path remains a gated correctness
proof.

## Use Cases

- Start the DS4 V100 appliance from a stable config file instead of an ad hoc
  command line.
- Validate model, pack index, GPU visibility, port, context, slot, and reserve
  settings before the expensive resident upload begins.
- Supervise the service under systemd or Kubernetes with documented restart
  behavior.
- Probe `/health`, `/v100/status`, `/metrics`, and generation through a
  production-like launcher path.
- Roll back to the base non-MTP service by using the default deployment mode,
  because speculative serving is not yet exposed.

## Architecture

- Keep the topology in `docs/architecture/DS4-V100-LAYOUT.md`: eight visible
  V100s, contiguous stage ownership, gpu7 output head, pure resident weights,
  and one active slot.
- Add a shell launcher that is the deployment contract:
  - loads an env file;
  - validates required files and numeric config;
  - checks GPU visibility and minimum free-memory reserve when `nvidia-smi` is
    available;
  - prints the exact server command;
  - execs `tools/ds4-v100-replay --serve`.
- Keep server status truthful:
  - readiness level remains `2` for the served base one-slot endpoint;
  - `mtp_enabled=false`;
  - limits expose one slot, sequential requests, no streaming, and max tokens;
  - `/metrics` exposes basic process counters and configured limits.
- Add deployment artifacts:
  - env example for operator-owned config;
  - systemd unit template;
  - Kubernetes pod/service manifest for the `gpu-01`/`llm` convention;
  - operations runbook updates.
- Add a production deployment smoke that starts the launcher, probes health,
  status, metrics, and one generation request, then feeds the result into the
  full gate as `production_deployment`.

## Parallel Work

The sprint supports parallel sidecar agents for:

- cluster/deployment convention review from prior sprint handoffs;
- status/metrics/gate shape review without touching model math;
- cluster validation after the local launcher and shell checks pass.

## Implementation

1. Add `tools/ds4-v100-run-appliance.sh`.
2. Add deployment config and supervisor artifacts under `deploy/v100/`.
3. Add `/metrics` and richer limits/status fields to `tools/ds4-v100-replay.c`.
4. Add `tools/ds4-v100-production-deployment-gate.sh`.
5. Wire the production deployment gate into `tools/ds4-v100-gate.sh`.
6. Update `docs/operations/DS4-V100-APPLIANCE.md`.
7. Update the vision/readiness ladder so production deployment can close while
   throughput optimization remains the next honest blocker.

## Files Summary

- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-production-deployment-gate.sh`
- `tools/ds4-v100-replay.c`
- `tools/ds4-v100-gate.sh`
- `tools/ds4-v100-appliance-smoke.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `deploy/v100/ds4-v100-appliance.service`
- `deploy/v100/ds4-v100-appliance.k8s.yaml`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `docs/sprints/SPRINT-043-REPORT.md`
- `docs/sprints/SPRINT-043-FOLLOWUPS.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local shell syntax checks pass for all new/changed shell scripts.
- Local deployment config check passes against the example env without needing
  model files.
- `tools/ds4-v100-replay --help` documents `/metrics`.
- The V100 cluster builds `tools/ds4-v100-replay`.
- The production deployment smoke starts through
  `tools/ds4-v100-run-appliance.sh` on the 8x V100 pod, probes health/status/
  metrics, and verifies the official fixture first token bytes `3136`.
- The full V100 gate includes `production_deployment PASS`, no longer reports
  `missing=production_deployment`, and reports the next remaining blocker
  honestly.
- Sprint report records commands, outputs, artifacts, and remaining gaps.
- Vision document and operations runbook reflect the shipped package and the
  next readiness step.

## Risks

- Adding another full service-start smoke lengthens gate runtime because each
  process performs resident upload. Keep the smoke short and bounded.
- The deployment package is only as strong as the current sequential one-slot
  endpoint. Avoid implying concurrency, streaming, external exposure, or MTP
  speculative serving.
- Restart behavior is operationally correct but expensive until stage
  open/upload is parallelized or made persistent across restarts.

## Security

Default host remains `127.0.0.1`. External exposure, auth, OpenAI-compatible
facade, multi-tenant isolation, and concurrent request handling remain
explicitly out of scope.

## Dependencies

- Sprint 032 base appliance health/status/repeated-request smoke.
- Sprint 042 MTP correctness gate.
- `docs/architecture/DS4-V100-LAYOUT.md`.
- Real base model, MTP sidecar, pack index, and 8x V100 cluster access.

## Open Questions

- Should the first production-managed service be systemd on the host or a
  Kubernetes pod in namespace `llm`? This sprint ships both templates and uses
  the launcher as the common contract.
- Should throughput optimization focus first on parallel stage upload,
  multi-token resident decode baselines, or multi-slot admission?
