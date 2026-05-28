# Sprint 421: Rank-Local Attention Projection HTTP Promotion

Date: 2026-05-27

## Objective

Promote Sprint 416's rank-local attention projection input from direct-decode
evidence into the HTTP serving harness, while keeping the work TP/EP-only.

## Scope

In scope:

- HTTP selected-token A/B at a serving-shaped long-context configuration.
- First-token and response-count parity.
- Decode, wall, compressed-KV, GPU-util, and VRAM summary comparison.
- Use the permanent profile harness and artifact layout.

Out of scope:

- PP/layer-split work.
- MTP.
- Making the gate a production default before larger chat/readiness runs.
- Solving the expert-residency memory issue discovered in Sprint 416.

## Configuration

Initial validation shape:

```text
endpoint=selected-token
requests=8
slots=8
ctx=262144
position=262080
tokens/request=8
hc-current-nccl=on
persistent graph replay=on
defer-nccl-init=on
tp-runtime-scratch=256 MiB
skip unused comp-state=on
model-router routes=on
compact MoE=on
```

## Definition of Done

- Control HTTP run returns all responses.
- Rank-local HTTP run returns all responses.
- First selected token matches.
- Status decode throughput improves or the blocker is documented.
- VRAM admission and NCCL reserve failures are recorded.
- GPU utilization is sampled.
- Results are written to a TEMP status report and reflected in `VISION.md`.

## Outcome

Status: HTTP selected-token gate complete at 8 and 28 slots.

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/control-selected-slot8-token8-container/
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/ranklocal-selected-slot8-token8-container/
```

Results:

| Metric | Control | Rank-local |
|---|---:|---:|
| HTTP 200 | 8/8 | 8/8 |
| First token | 45124 | 45124 |
| Client generated tok/s | 22.180780 | 24.225369 |
| Status generated decode tok/s | 88.402819 | 100.059560 |
| Status continuation decode tok/s | 94.811395 | 107.260053 |
| Scaffold projected slot-step tok/s | 94.272991 | 108.583858 |
| Avg sampled GPU util | 6.95% | 6.85% |
| Min free VRAM | 6886 MiB | 6886 MiB |
| NCCL reserve failures | 0 | 0 |

28-slot follow-up:

| Metric | Control | Rank-local |
|---|---:|---:|
| HTTP 200 | 28/28 | 28/28 |
| First token | 45124 | 45124 |
| Client generated tok/s | 52.690208 | 59.592035 |
| Status generated decode tok/s | 129.750653 | 158.385152 |
| Status continuation decode tok/s | 131.436685 | 162.101543 |
| Avg sampled GPU util | 16.328125% | 13.696429% |
| Min free VRAM | 4570 MiB | 4570 MiB |
| NCCL reserve failures | 0 | 0 |

The first 28-slot rank-local candidate was externally terminated during
readiness with `rc=-15`; the detached retry completed cleanly and is the result
recorded above.

Decision:

- Keep `--true-ds4-attention-projection-rank-local-input-gate` as the next
  serving promotion candidate.
- Do not yet enable it as a production default from selected-token checks alone.
- Next validation should move to chat/readiness/parity before default
  promotion.

Detailed report:

```text
TEMP_STATUS_REPORT_421.md
```
