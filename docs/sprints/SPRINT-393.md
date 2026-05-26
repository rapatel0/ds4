# Sprint 393: TP/EP Serving Readiness Gate

## Overview

Turn the current TP/EP serving evidence into a reusable readiness gate before
continuing deeper performance work.

Sprint 392 proved multi-prompt HTTP soak and parity comparison, but the serving
artifact validation is still spread across manual summary inspection, parity
JSON, VRAM fields, GPU-util fields, and response metadata. This sprint adds a
single permanent checker so future throughput gates cannot be promoted unless
the target `32` slot / `256K` serving shape remains operational.

## Scope

- Add a standalone HTTP serving readiness checker for one profile artifact
  case.
- Validate response artifacts, `summary.json`, `status.json`, and optional
  `gpu_util.csv` / metrics files.
- Enforce the target TP/EP invariants:
  - `32` configured slots and `256K` context when requested.
  - all responses HTTP 200.
  - generated token sequences are present and have the requested length.
  - `token_match` stays clean when present.
  - KV runtime is resident and HC persistence is enabled.
  - typed DS4 KV gates are enabled.
  - VRAM admission reports zero failures and minimum free memory above
    threshold.
  - GPU sampling exists when requested.
  - prompt-file soak metadata matches the expected prompt count/digest when
    requested.
- Emit a machine-readable readiness summary JSON for sprint reports.
- Validate against existing Sprint 392 V100 artifacts and a negative fixture.

## Out Of Scope

- No PP/layer-split work.
- No MTP integration.
- No new kernel implementation.
- No promotion of E5M2 or any other default.
- No redefinition of production readiness; this is a gate that makes current
  serving evidence explicit, not the final performance solution.

## Definition Of Done

- `tools/ds4-v100-http-readiness-check.py` exists and is syntax checked.
- The checker passes on a real V100 TP/EP `32` slot / `256K` multi-prompt HTTP
  artifact from Sprint 392.
- The checker fails non-zero on a mutated/invalid artifact.
- `docs/sprints/VISION.md` and `docs/sprints/STATUS.md` describe the gate and
  how it fits the throughput plan.
- `TEMP_STATUS_REPORT_393.md` records the validation result and current
  topline.
- The sprint document records outcome and decision.

## Risks

- A too-loose gate lets broken serving runs look usable. The checker should
  validate response-level metadata, not only aggregate summary fields.
- A too-tight gate can block diagnostics. Thresholds must be configurable so
  experiments can run with narrower checks while production promotion keeps the
  full target shape.

## Execution Plan

1. Implement the readiness checker with explicit CLI thresholds.
2. Validate it on Sprint 392 control artifacts.
3. Create a negative fixture and confirm non-zero failure.
4. Update vision/status/report with the gate and current topline metrics.
5. Commit the kept artifacts.

## Outcome

Complete. Added `tools/ds4-v100-http-readiness-check.py` as a permanent
single-case HTTP serving artifact gate.

The checker validates response files plus `summary.json` / `status.json` and
can enforce the target TP/EP serving invariants: HTTP 200 count, response
count, generated-token length, `32` slots, `256K` context, prompt soak
metadata, resident KV/HC state, typed DS4 KV gates, compact MoE, token-match
metadata, DS4 checksums, GPU sampling, and VRAM admission.

## Validation

Local syntax:

```text
python3 -m py_compile tools/ds4-v100-http-readiness-check.py
```

Real V100 artifact readiness, Sprint 392 control:

| Metric | Value |
|---|---:|
| Ready | `true` |
| Failures | `0` |
| HTTP 200 | `32` |
| Tokens/request | `32` |
| Server decode tok/s | `106.390802` |
| Client generated tok/s | `38.912861` |
| Avg GPU util | `9.772727%` |
| Max GPU util | `52%` |
| First token | `83484` |
| VRAM failures | `0` |
| Min free VRAM | `1746 MiB` |

Real V100 artifact readiness, Sprint 392 E5M2 candidate:

| Metric | Value |
|---|---:|
| Ready | `true` |
| Failures | `0` |
| HTTP 200 | `32` |
| Tokens/request | `32` |
| Server decode tok/s | `106.483285` |
| Client generated tok/s | `39.774181` |
| Avg GPU util | `9.691860%` |
| Max GPU util | `50%` |
| First token | `83484` |
| VRAM failures | `0` |
| Min free VRAM | `1746 MiB` |

Negative fixture:

```text
ready=false
failure_count=2
failed_checks=["token_match", "checksum"]
```

## Decision

Promote the readiness checker as the standard artifact gate for future TP/EP
serving promotions. This sprint does not promote a performance gate; it makes
the evidence contract explicit so the next throughput sprint can change
steady-state scheduling or kernels without weakening serving correctness,
VRAM, or multi-prompt coverage.

## Artifacts

- `logs/from-cluster/sprint392-multiprompt-e5m2/http-control/readiness-summary.json`
- `logs/from-cluster/sprint392-multiprompt-e5m2/http-candidate-e5m2/readiness-summary.json`
