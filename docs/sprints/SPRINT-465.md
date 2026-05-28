# Sprint 465: TP/EP Graph Event-Order Boundary Triage

## Objective

Determine whether the graph no-replay serving failure is caused by the
output-head boundary or by earlier graph-event-ordered decode state.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Keep all graph work default-off.
- Add permanent diagnostic controls only when they help future graph triage.

## Implementation

Code changes:

- Added graph-order event rings to avoid repeatedly recording the same CUDA
  event objects inside one graph/event-ordered decode step.
- Added graph-mode output-head boundary waits before gathering
  `d_final_hc_shard`.
- Added diagnostic `--decode-cudagraph-output-sync-gate` plus launcher/profile
  A/B wiring:
  - `DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC`
  - `--decode-cudagraph-output-sync`
  - `--candidate-decode-cudagraph-output-sync`
  - `--control-decode-cudagraph-output-sync`

## Validation

All builds and harness checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Event-ring A/B:

```text
/localpool/ds4/workspace/logs/s462-event-ring-graph-gate-s8-t3
```

Output-head rank/dense boundary A/B:

```text
/localpool/ds4/workspace/logs/s463-output-head-rank-dense-boundary-s8-t3
```

Output-head full device-sync diagnostic A/B:

```text
/localpool/ds4/workspace/logs/s464-graph-output-sync-s8-t3-r2
```

| Candidate | First Token | Parity | Server Decode Tok/s | HC-Current Gather ms | Decision |
|---|---:|---:|---:|---:|---|
| control baseline | 52762 | pass | 20.638517 | 4.398480 | baseline |
| event ring, no replay | 57097 | 0/8 | 9.328611 | 157.328098 | reject |
| output rank+dense wait | 42549 | 0/8 | 9.418328 | 160.449560 | reject |
| output full device sync | 42549 | 0/8 | 9.088940 | 187.396993 | reject |

## Decision

Do not promote graph-event-order serving. Full device synchronization before
the output head still fails parity, so the wrong state is produced before the
output head reads it. The output-head boundary was incomplete but not the root
cause.

## Next

Add first-divergence instrumentation inside the decode step. The next sprint
should emit checksums after major stages under eager and graph-event-order:

- HC-current input
- attention projection
- compressed KV/state update
- raw/window attention read
- attention output projection
- post-attention FFN input
- routed FFN/compose
- final HC expansion

The goal is to identify the first stage where graph-event-order diverges, not
to run another broad serving A/B.
