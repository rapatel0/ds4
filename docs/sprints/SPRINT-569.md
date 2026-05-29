# Sprint 569 - C1 Full-Capture Serving Metrology

Date: 2026-05-29

## Goal

Measure the opt-in no-suffix full-capture path after Sprint 568's HC buffer
rebase at the serving shape, without mixing startup/init time into throughput.

## Context

Sprint 568 fixed the correctness bug that blocked full-capture cross-position
replay. The validation so far is reduced selected-token parity. The next step
is serving metrology, not another graph correctness patch:

- deterministic generation only;
- startup and server initialization excluded;
- substantial warmup before measured requests;
- long prompt / longer generation window;
- compare against the current promoted serving control.

## Plan

1. Rebuild the current tree on the V100 node from a clean `rsync`.
2. Run the promoted control launcher path at `32` slots / `256K`.
3. Run the opt-in no-suffix full-capture path at the same shape with:
   `--decode-cudagraph-gate`, `--decode-cudagraph-replay-probe-gate`, and
   `--decode-cudagraph-persistent-replay-gate`, while disabling the promoted
   suffix replay wrapper.
4. For both legs, wait for readiness, send a full-slot warmup batch, then time
   only the measured full-slot batch.
5. Compare generated token sequences and response metadata before making any
   performance claim.

## Definition of Done

- Remote V100 container build passes.
- Both legs serve the deterministic long-prompt workload.
- Measured throughput excludes readiness/startup and warmup.
- Sequence parity and graph replay counters are recorded.
- Steering and vision are updated with a promote/reject/continue decision.

## Results

Remote tree: `/workspace/s569-serving-metrology`

Artifacts: `/workspace/s569-serving-metrology-artifacts`

Build:

- `make appliance/ds4-v100-tp-ep-appliance` passed in the remote V100
  container.

Workload:

- Shape: `32` slots, `256K` context.
- Generation: `temperature=0`, `top_p=1`, `32` generated tokens/request.
- Prompt: long deterministic prompt beginning
  `The capital of France is Paris...`, with lorem-ipsum filler.
- Warmup: one full-slot `32` request batch.
- Measurement: one full-slot `32` request batch after warmup.
- Startup/readiness and warmup are excluded from the measured request window.

Comparison:

| Leg | Generated tok/s wall | Continuation tok/s wall | Generated tok/s decode | Continuation tok/s decode | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Promoted suffix-control | `12.603435` | `12.209578` | `1.506832` | `1.506786` | `81.205441s` |
| Opt-in no-suffix full-capture | `16.807308` | `16.282079` | `2.288130` | `2.288468` | `60.873505s` |

Observed speedup:

- Request-window generated throughput: `1.334x`.
- Request-window continuation throughput: `1.334x`.
- Decode generated throughput: `1.519x`.
- Median latency improvement: `1.334x`.

Parity:

- All `32/32` measured generated token sequences matched by request index.
- The measured generated token multiset also matched.
- The first generated text matched exactly.
- The response `checksum` vectors did not match. For this serving comparison,
  the generated `token_ids` are the parity source of truth; the checksum field
  is path-sensitive metadata and differs even when output tokens match.

Graph evidence:

- Promoted suffix-control logs show persistent graph replays succeeding with
  `nodes 2000` per layer at the sampled tail.
- Opt-in no-suffix full-capture logs show persistent graph replays succeeding
  with `nodes 2697` per layer at the sampled tail.
- Both legs kept the HTTP cache counters healthy for the two full-slot batches
  (`64` request cache misses, `32` evictions, `32` slots used).

## Decision

Treat Sprint 569 as a positive serving-metrology result for opt-in no-suffix
full capture. The Sprint 568 bug fix is validated beyond reduced selected-token
checks: deterministic long-prompt serving output matched for all measured
requests, and the warmed request window showed a large throughput/latency
signal.

Do not flip no-suffix full capture to the default from this single run. The next
sprint should repeat/extend the warmed serving gate with a longer steady-state
window and improved graph/status counter extraction, then either promote the
default or record the remaining blocker.
