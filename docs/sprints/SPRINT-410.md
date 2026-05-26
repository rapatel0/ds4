# Sprint 410: HC-Current NCCL HTTP A/B Promotion Gate

## Goal

Decide whether the now memory-admitted HC-current NCCL allgather path should be
promoted beyond diagnostic status at the production TP/EP serving shape:

```text
32 slots
256K context
32 generated tokens/request
32 concurrent HTTP requests
lazy output head
compact MoE decode
model-router routes
skip unused TP-runtime comp-state
```

Sprint 409 proved the memory gate. This sprint is the throughput/default
decision.

## Implementation

Add `tools/ds4-v100-tp-ep-nccl-http-ab.py`, a permanent harness that composes
the existing tools instead of duplicating server logic:

- run non-NCCL HTTP control through `tools/ds4-v100-tp-ep-profile.py`
- run HC-current NCCL candidate through the same profile path with
  `--hc-current-stream-sync --hc-current-nccl-allgather`
- validate each case with `tools/ds4-v100-http-readiness-check.py`
- compare response artifacts with `tools/ds4-v100-http-response-parity.py`
- write `ab-summary.json` and `ab-summary.md`

The harness treats correctness/readiness separately from promotion. A correct
candidate that is flat or slower remains useful evidence and exits cleanly if
all validation passes.

## Definition of Done

- [x] Local `py_compile` passes for the new harness.
- [x] V100 target-shape HTTP A/B completes for both control and candidate.
- [x] Both cases pass HTTP readiness with GPU samples, resident KV, typed KV,
      compact MoE, checksum, token-match, and VRAM admission checks.
- [x] Response parity matches across all request artifacts.
- [x] `ab-summary.json` and `ab-summary.md` record the topline metrics and
      promotion decision.
- [x] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and a temporary
      status report are updated with the measured result.

## Promotion Rule

Promote HC-current NCCL only if:

- both control and candidate pass readiness,
- response parity matches,
- target-shape VRAM reserve has zero failures,
- candidate server generated decode tok/s improves by at least 2%.

Otherwise keep HC-current NCCL default-off and diagnostic-only, then move the
NCCL work toward broader TP/EP collectives rather than this narrow boundary.

## Outcome

V100 target-shape run:

```text
artifact: logs/from-cluster/sprint410-nccl-http-ab/
shape: 32 requests, 32 slots, 262144 ctx, 32 generated tokens/request
```

| Metric | Control | HC-current NCCL | Ratio |
|---|---:|---:|---:|
| server generated decode tok/s | 101.897890 | 107.723452 | 1.057171 |
| server continuation decode tok/s | 101.682616 | 107.545644 | 1.057660 |
| client generated tok/s | 17.223947 | 16.627120 | 0.965349 |
| avg GPU util % | 4.535714 | 3.524272 | 0.777005 |
| min free VRAM MiB | 2738 | 2106 | 0.769175 |
| HC-current gather ms | 3.279789 | 5.700894 | 1.738189 |
| HC-current input ms | 265.428902 | 254.759223 | 0.959802 |

Both cases returned `32/32` HTTP 200. Readiness passed for both cases with
resident KV, typed KV, compact MoE, checksums, token-match, GPU samples, and
zero VRAM failures. Response parity matched `32/32` request artifacts, and the
first output token was `83484` in both cases.

Decision: promote HC-current NCCL as the appliance default for TP/EP serving.
The server-side decode improvement clears the 2% promotion rule. The client
throughput and average utilization regressions remain tracked as operational
metrology caveats; they do not block promotion because the candidate is
correct, memory-admitted, and improves measured server decode work.
