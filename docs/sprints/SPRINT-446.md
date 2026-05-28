# Sprint 446: Rank-Major Gate Isolation

## Objective

Stay TP/EP only and isolate the correctness failure from Sprint 445's combined
rank-major serving candidate.

Sprint 445 proved the combined rank-major attention/FFN/router candidate has a
real speed signal but fails response parity. This sprint runs the same serving
shape with one candidate gate enabled at a time so the next implementation step
targets the actual divergent consumer instead of guessing.

## Implementation Plan

Run three same-binary HTTP A/Bs at the Sprint 445 shape:

- `8` requests;
- `8` slots;
- `256K` context;
- `2` generated tokens/request;
- chat endpoint;
- control: HC-current NCCL, post-attention FFN input, fixed-capacity route plan.

Candidate variants:

1. Attention input only:
   - `--candidate-attention-projection-rank-local-input`
2. FFN input only:
   - `--candidate-routed-ffn-rank-major-input`
3. Router logits only:
   - `--candidate-model-router-rank-major-logits`

## Validation

- Each A/B must produce control and candidate summaries.
- Readiness must pass for both legs.
- Response parity must be recorded for every variant.
- No stale DS4 GPU process may remain after each run.

## Execution Notes

The first attention-only run at `8` slots hit CUDA OOM during the control
expert-residency phase before a summary was written. The A/B harness still
launched the candidate, which is incorrect for decision experiments. The
harness now aborts immediately if either profile leg returns nonzero.

Isolation may use a smaller slot count when the goal is parity attribution,
because the question is which rank-major gate changes tokens, not target-shape
throughput promotion.

## Decision Rule

- If a single gate passes parity and improves server decode by at least `1.02x`,
  promote that gate or schedule a target-shape rerun.
- If a single gate fails parity, keep it diagnostic-only and inspect the
  corresponding consumer path.
- If all single gates pass but the combined gate fails, the bug is an
  interaction between rank-major buffers/scratch ownership and should be fixed
  before more performance work.

## Outcome

Artifacts:

- Attention-only:
  `/localpool/ds4/workspace/logs/s446-rankmajor-isolate-attn-s512`
- FFN-only:
  `/localpool/ds4/workspace/logs/s446-rankmajor-isolate-ffn-s512`
- Router-only:
  `/localpool/ds4/workspace/logs/s446-rankmajor-isolate-router-s512`

The final isolation runs used `8` slots, `4` requests, `256K` context, `2`
tokens/request, `512 MiB` TP runtime scratch, and deferred NCCL. The smaller
request count was used for correctness attribution after the initial `8`
request attention-only run hit expert-residency OOM before summary.

| Candidate gate | Parity | Control first token | Candidate first token | Control server tok/s | Candidate server tok/s | Speedup | Readiness |
|---|---:|---:|---:|---:|---:|---:|---|
| Attention input only | 0/4 | 72960 | 81401 | 20.379020 | 20.750186 | 1.018x | false due client threshold in reduced run |
| FFN input only | 4/4 | 72960 | 72960 | 19.850119 | 20.059547 | 1.011x | pass |
| Router logits only | 4/4 | 72960 | 72960 | 20.124833 | 20.449131 | 1.016x | pass |

## Decision

The correctness blocker is the attention projection rank-local/rank-major input
path. It changes the selected token by itself, so the combined Sprint 445
candidate failure is not an FFN/router interaction.

Keep the attention rank-local/rank-major input gate diagnostic-only and fix that
consumer before combining rank-major gates again.

The FFN-input and router-logits gates are correctness-clean at the reduced
isolation shape, but neither crosses the normal `1.02x` promotion threshold
there. They should remain default-off until the attention path is fixed and a
target-shape same-binary A/B proves the composed path.
