# Sprint 464: Promote TP/EP Parallel Expert Load

## Overview

Promote the validated TP/EP parallel expert residency loader into the appliance
startup default.

Sprint 463 showed the old startup path loaded expert residency in a visible
GPU-by-GPU round-robin pattern. The parallel loader fans out per-GPU expert
binding loads within each layer and removes that serial startup behavior. It
does not change steady-state decode throughput, but it materially improves
operator iteration time and prevents startup from polluting utilization
observations.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- No runtime decode optimization in this sprint.
- Keep the gate opt-out capable.
- Update production launcher defaults and the deployment env example.

## Definition of Done

- `DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD` defaults to enabled in the appliance
  launcher.
- The launcher validates `0/1/true/false/on/off` values for the env var.
- `--print-command` includes `--parallel-expert-load-gate` by default.
- The deployment env example documents the startup default and opt-out.
- The profile and HTTP A/B wrappers default to parallel expert load, with an
  explicit `--disable-parallel-expert-load` serial-load diagnostic.
- Local syntax/diff checks pass.
- The sprint decision and evidence are recorded.

## Evidence From Sprint 463

Clean 8-slot smoke:

```text
/localpool/ds4/workspace/logs/s460-parallel-expert-load-s8-t3-clean
```

| Metric | Result |
|---|---:|
| HTTP responses | 8/8 |
| readiness elapsed | 53.110685 s |
| request elapsed | 11.508914 s |
| server generated decode tok/s | 20.688739 |
| min free VRAM | 5092 MiB |
| VRAM failures | 0 |

Target-shape long run:

```text
/localpool/ds4/workspace/logs/s460-parallel-expert-load-s32-t32-long
```

| Metric | Result |
|---|---:|
| shape | 32 requests / 32 slots / 256K / 32 tokens |
| HTTP responses | 32/32 |
| generated tokens | 1024 |
| readiness elapsed | 106.215634 s |
| request elapsed | 67.038542 s |
| server generated decode tok/s | 35.813083 |
| request-window avg GPU util | 12.534426% |
| min free VRAM | 1734 MiB |
| VRAM failures | 0 |

## Implementation

Updated:

- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`
- `deploy/v100/ds4-v100-appliance.env.example`

The binary flag remains:

```text
--parallel-expert-load-gate
```

The launcher env remains:

```text
DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD
```

The wrapper opt-out flag is:

```text
--disable-parallel-expert-load
```

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check -- tools/ds4-v100-run-appliance.sh deploy/v100/ds4-v100-appliance.env.example docs/sprints/SPRINT-464.md docs/sprints/VISION.md TEMP_STATUS_REPORT_464.md
```

Launcher command checks:

```text
DS4_V100_SERVE_MODE=tp-ep tools/ds4-v100-run-appliance.sh --print-command
DS4_V100_SERVE_MODE=tp-ep DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=0 tools/ds4-v100-run-appliance.sh --print-command
```

## Decision

Promote parallel expert loading as the TP/EP production startup default.

This is not a steady-state throughput optimization. The active throughput work
remains HC-current staging and routed FFN/EP cost reduction.
