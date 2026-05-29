# Sprint 570 - C1 No-Suffix Full-Capture Promotion Gate

Date: 2026-05-29

## Goal

Decide whether opt-in no-suffix full capture is ready to become the TP/EP
launcher default.

Sprint 569 produced a strong serving-metrology signal, but it used one measured
full-slot batch. This sprint repeats the same comparison with a longer
steady-state measured window, explicit graph/status extraction, deterministic
generation, and startup/init excluded.

## Context

Authoritative order:

- `SPIKE_B_STEERING.md`: next item is `C1 longer steady-state serving promotion gate`.
- `docs/sprints/VISION.md`: Sprint 569 is a positive opt-in signal, not a
  default flip.

Current control:

- The promoted TP/EP launcher default is graph suffix replay:
  `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=1`.

Candidate:

- Disable suffix replay and enable no-suffix full capture:
  `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=0`
- Pass:
  `--decode-cudagraph-gate`
  `--decode-cudagraph-replay-probe-gate`
  `--decode-cudagraph-persistent-replay-gate`

## Plan

1. Rebuild the current tree on the V100 node from a clean `rsync`.
2. Run the promoted suffix-control leg at `32` slots / `256K`.
3. Run the opt-in no-suffix full-capture leg at the same shape.
4. For each leg:
   - wait for readiness;
   - run at least one full-slot warmup batch;
   - time only a longer measured request window;
   - use deterministic generation (`temperature=0`, `top_p=1`);
   - use a long prompt to keep the generation path representative;
   - record `/status`, `/metrics`, response artifacts, server logs, and parsed
     graph replay counters.
5. Compare generated token sequences by request index and multiset.
6. Compare request-window throughput, response metadata timing, graph replay
   success, invalidations, peer-copy/SYS counters, and VRAM failures.

## Promotion Criteria

Promote no-suffix full capture as the launcher default only if all are true:

- Remote V100 build passes.
- Both legs complete the same measured workload with all HTTP 200 responses.
- Measured generated token sequences match for every request, and the measured
  sequence multiset also matches.
- The candidate has successful full-capture graph replays after warmup and no
  graph invalidation pattern that would make the measured window misleading.
- Peer-copy/SYS and NCCL graph SYS counters remain clean.
- Startup/init and warmup are excluded from the throughput claim.
- Candidate request-window generated tok/s and median latency are materially
  better than the promoted suffix-control leg.

## Rejection / Continue Criteria

Do not promote if generated token sequences diverge, graph replay does not
actually occur in the measured window, the candidate regresses materially, or
the counters show hidden transport/SYS regressions.

If parity and performance pass but the evidence is still too narrow, record the
candidate as correctness-clean and performance-positive, then run one additional
promotion gate rather than changing defaults.

## Definition of Done

- Sprint 570 artifact paths are recorded.
- The run script/harness records excluded readiness and warmup duration.
- The measured comparison uses at least `128` measured requests or an equivalent
  longer steady-state window.
- Sequence parity is verified across all measured responses, not only the first.
- Graph/status counters are parsed from logs or status artifacts.
- Steering and vision are updated with a promote/reject/continue decision.
- All repo changes from this sprint are committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Results

Remote tree: `/workspace/s570-full-capture-promotion`

Artifacts: `/workspace/s570-full-capture-promotion-artifacts`

Build:

- `make appliance/ds4-v100-tp-ep-appliance` passed in the remote V100
  container.

Workload:

- Shape: `32` slots, `256K` context.
- Generation: `temperature=0`, `top_p=1`, `64` generated tokens/request.
- Warmup: `2` full-slot batches, `64` requests total, excluded.
- Measurement: `4` full-slot batches, `128` requests total, timed.
- Prompt: long deterministic prompt beginning
  `The capital of France is Paris...`.

Measured performance:

| Leg | HTTP 200 | Continuation tok/s wall | Continuation tok/s decode | Median latency | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promoted suffix-control | `128/128` | `16.618822` | `1.513915` | `132.083580s` | `132.632103s` |
| Opt-in no-suffix full-capture | `128/128` | `20.814267` | `2.247000` | `103.958728s` | `105.384322s` |

Candidate speedup:

- Continuation request-window throughput: `1.252x`.
- Continuation decode throughput: `1.484x`.
- Median latency: `1.271x`.

Graph / transport counters:

| Leg | Persistent lines | Cache-hit lines | Cache-miss lines | Replay-succeeded lines | Tail graph nodes | Peer-copy hits | NCCL SYS hits |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Promoted suffix-control | `28724` | `28724` | `0` | `28724` | `[2000]` | `0` | `0` |
| Opt-in no-suffix full-capture | `28724` | `28724` | `0` | `28724` | `[2697]` | `0` | `0` |

Parity:

- Generated token metadata was available as
  `ds4_v100.generated_token_sequence`.
- Generated token sequences diverged for `128/128` measured responses.
- Generated text diverged for `128/128` measured responses.
- First differences are often immediate:
  - measured `0`: control `98649`, full capture `26838`;
  - measured `31`: control `98649`, full capture `79468`;
  - measured `32`: control `123327`, full capture `98649`;
  - measured `64`: control `79468`, full capture `98649`.

Diagnostic notes:

- The first harness summary incorrectly reported empty sequence parity because
  it read `choices[0].message.token_ids`; in this response shape the token ids
  live in `choices[0].token_ids` and `ds4_v100.generated_token_sequence`.
  Direct artifact parsing above is the authoritative Sprint 570 parity result.
- Both legs replayed graphs with no cache misses in the measured window and no
  peer-copy/SYS transport hits.
- Response metadata shows prompt-prefill/cache behavior differs across batches
  and legs. Example measured response `0`: control reported
  `batch_prompt_tokens=203`, while full capture reported
  `batch_prompt_tokens=32`. This may be a useful localization clue, but it does
  not change the promotion decision because generated output diverged.

## Decision

Reject default promotion of no-suffix full capture.

Sprint 570 confirms the performance ceiling is real, but correctness does not
hold for the longer `64` token / `128` measured request serving gate. The next
sprint should localize why Sprint 569's `32` token gate matched while Sprint
570 diverges, with special attention to long-generation replay state,
prompt-cache/coalescing state, and cache-hit replay after multiple full-slot
warmup batches.
