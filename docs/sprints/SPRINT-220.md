# Sprint 220 - Production Deployment Defaults For Warmed 256K Serving

Date: 2026-05-23
Status: Planned

## Overview

Sprint 219 validated the warmed `32`-slot/`256K` production serving mode and
made warmup readiness visible through `/v100/status` and `/metrics`. The next
gap is operator packaging: the example deployment environment and the older
production deployment smoke still reflect pre-Sprint-218 assumptions such as
`1M`/small-slot defaults, source-style pack-index wiring, and a `16`-slot
deployment gate cap.

Sprint 220 updates the operator-facing deployment path so the shipped appliance
configuration matches the validated production mode.

## Goals

- Update `deploy/v100/ds4-v100-appliance.env.example` to the warmed
  `ctx=262144`, `slots=32`, `active_microbatch=32` production appliance mode.
- Update the production deployment gate so it supports `--appliance-dir`,
  `32` slots at `256K`, startup warmup, and warmed-readiness checks.
- Run the updated deployment gate on the V100 pod against the production pack.
- Keep the benchmark/throughput gate separate: deployment smoke proves launch,
  readiness, health/status/metrics, and a bounded generation request; Sprint
  219 remains the throughput proof.

## Non-Goals

- No new kernel tuning.
- No TP scheduler work.
- No MTP throughput work.
- No long throughput re-run unless the deployment smoke exposes a mismatch.

## Implementation

1. Refresh `deploy/v100/ds4-v100-appliance.env.example`:
   - production appliance dir set to the current pack path;
   - context `262144`;
   - slots and active microbatch `32`;
   - startup warmup `auto`;
   - production TurboMind/F8 flags aligned with the Sprint 219 gate;
   - comments updated so 256K/32 is no longer described as rejected.
2. Update `tools/ds4-v100-production-deployment-gate.sh`:
   - accept `--appliance-dir DIR`;
   - remove the stale `slots <= 16` local guard and rely on launcher admission;
   - add `--startup-warmup auto|0|1`;
   - validate `warmup_required`/`warmed_ready` in `/v100/status` for
     `ctx=262144`, `active_microbatch>16`;
   - validate `ds4_v100_warmup_required` and `ds4_v100_warmed_ready` metrics
     for that shape.
3. Run the updated deployment gate on the cluster:

   ```text
   ctx=262144
   slots=32
   active_microbatch=32
   tokens=2
   requests=2
   startup_warmup=auto
   ```

4. Copy logs to `logs/from-cluster/sprint220-production-deployment-warmed/`.
5. Update sprint/status/vision/runbook docs and commit.

## Files In Scope

| File | Purpose |
|---|---|
| `deploy/v100/ds4-v100-appliance.env.example` | production operator defaults |
| `tools/ds4-v100-production-deployment-gate.sh` | deployment smoke |
| `docs/sprints/SPRINT-220.md` | plan, evidence, decision |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | vision progress |
| `docs/operations/DS4-V100-APPLIANCE.md` | operator command/runbook |
| `logs/from-cluster/sprint220-production-deployment-warmed/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] Env example defaults to warmed `32`-slot/`256K` appliance serving.
- [ ] Deployment gate accepts appliance dirs and `32` slots at `256K` through
      launcher admission.
- [ ] Deployment gate validates warmed readiness in status and metrics.
- [ ] Local shell validation passes.
- [ ] V100 deployment smoke passes with the production pack.
- [ ] Logs are copied to
      `logs/from-cluster/sprint220-production-deployment-warmed/`.
- [ ] Docs are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
model: /models/DSv4-Flash-256e-fixed.gguf
```

Deployment smoke:

```text
./tools/ds4-v100-production-deployment-gate.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --ctx 262144 \
  --slots 32 \
  --active-microbatch 32 \
  --queue-policy sequential \
  --tokens 2 \
  --requests 2 \
  --startup-warmup auto
```

Expected:

- launcher check passes;
- health/status/metrics pass;
- status reports `warmup_required=true` and `warmed_ready=true`;
- metrics report `ds4_v100_warmup_required 1` and `ds4_v100_warmed_ready 1`;
- each request returns the expected first token.

## Risks

- Deployment smoke is not a throughput benchmark; do not use it as tok/s proof.
- The env example contains many historical diagnostic flags. Keep the edit
  scoped to validated production defaults and stale cap comments.
- The old source-layout pack-index path must continue working for legacy
  diagnostics, but production examples should prefer `--appliance-dir`.

## Security

No external exposure. Keep default host as `127.0.0.1`. Do not log model
weights.

## Dependencies

- Sprint 219 warmed readiness and 64-request production gate.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
