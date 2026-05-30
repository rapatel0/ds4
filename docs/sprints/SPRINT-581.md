# Sprint 581 - Tuning Sprint: Reference-Shape Decode Throughput

Date: 2026-05-29

## Goal

Now that A1-A4 and C1 (both suffix-replay and the now-default full capture) are
promoted, measure the de-confounded steady-state decode throughput at the
reference shape with the **promoted full-capture default**, and establish the new
baseline number that supersedes the pre-full-capture reference in
`SPIKE_B_STEERING.md` (~`35.9` tok/s server decode / ~`889` ms decode domain).

This is the ordered "tuning sprint" from the vision sequence. It **opts into perf
measurement** (per `VALIDATION_CONTROL_POLICY`).

## Scope

Phase 1 (this sprint): the headline reference-shape measurement and the new
baseline. Compare promoted full-capture default vs the prior suffix-control
reference at the de-confounded shape, decode-domain timing, and GPU util.

Deferred to follow-on tuning sub-sprints (scoped out here to keep the measurement
clean):
- Slots x context shape envelope sweep.
- NCCL ring / topology pinning.
- C4 KV spill at long context.

These are recorded in the steering tuning item; they are tuning levers that only
matter once the baseline is established and a binding constraint is identified.

## Reference shape (de-confounded steady-state)

- `32` slots / `256K` context, deterministic (`temperature=0`, `top_p=1`).
- `64` generated tokens/request, multiple full-slot measured batches
  (steady-state window), startup + warmup excluded.
- Promoted default leg: `DS4_V100_TP_EP_DECODE_GRAPH_MODE=full` (no env override).
- Reference leg: `DS4_V100_TP_EP_DECODE_GRAPH_MODE=suffix` (prior promoted).

## Plan

1. Reuse the Sprint 579/580 build (carries the promoted launcher + barrier fix).
2. Run the reference workload on the promoted full-capture default and on the
   suffix-control reference; exclude startup + warmup; time the measured window.
3. Record server decode tok/s, per-request continuation decode tok/s, median/P95
   latency, decode-domain ms, and graph replay / peer-SYS counters.
4. Update the steering reference numbers to the full-capture default baseline.

## Definition of Done

- Reference-shape decode throughput measured for the full-capture default and the
  suffix reference at the de-confounded shape; artifacts recorded.
- The steering reference baseline is updated to the full-capture default.
- Deferred tuning levers (shape envelope, NCCL pinning, C4 spill) recorded in the
  steering tuning item.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Results

Artifacts: `/workspace/s581-tuning-artifacts`, `/workspace/s581-eager-artifacts`.

### Reference-shape baseline (de-confounded, `32` slots / `256K` / `64` tok/req / `4` measured batches, `128` req)

| Mode | Agg decode tok/s | Per-req decode tok/s | Median latency | P95 |
| --- | ---: | ---: | ---: | ---: |
| `full` (promoted default) | `26.821` | `2.344` | `74.936s` | `75.852s` |
| `suffix` (prior default) | `21.889` | `1.529` | `92.087s` | `92.115s` |

Full-capture default speedup vs suffix: **`1.225x` aggregate decode, `1.53x`
per-request decode**, median latency `92.1s -> 74.9s`. This is the new
reference-shape baseline (supersedes the pre-full-capture regime). The new
`DS4_V100_TP_EP_DECODE_GRAPH_MODE` knob (`full`/`suffix`) was exercised in
production form and resolves correctly.

### Decode-domain gap attribution (eager leg, per-step `ms`, total `14.445`)

The per-stage timers populate only in pure eager (both graph modes launch the
captured region as one unit). Eager attribution:

| Stage | ms/step | % |
| --- | ---: | ---: |
| EP (MoE all-to-all) | `9.419` | `65.2%` |
| attention (proj+state+raw+output) | `1.774` | `12.4%` |
| compose (+reduce/copy/final) | `1.570` | `10.9%` |
| HC-current input | `1.096` | `7.6%` |
| final_hc | `0.520` | `3.6%` |
| host-sync (route_upload+fill_pack+router_select) | `0.749` | `~5.2%` |

**EP/MoE all-to-all is 65% of the decode step** -- the dominant cost and the
source of the "waves" (compute+exchange bursts separated by cross-GPU dispatch/
combine). This is the small-batch MoE regime: ~`0.75` tokens/expert at `32` slots
(top-6 over `256` experts) leaves grouped-GEMM tiles nearly empty (matching the
observed `SMOCC ~0.08-0.11`). Host-sync orchestration (the PCIe/GPU0 skew) is only
~`5%` of step time -- real but not the throughput bottleneck.

## Decision

New reference baseline recorded: `26.8` tok/s aggregate decode at the
de-confounded shape on the full-capture default (`1.225x` over suffix). The gap
attribution confirms the roadmap: the throughput headroom is in **EP** (65%), via
**B2** (dispatch/combine efficiency) and especially **MTP/B1** (which fills expert
tiles by making each step see `(K+1)x` tokens, attacking the root cause of the
65% EP cost and the low occupancy). Host-sync/output-head decentralization is a
~5% secondary cleanup, not the pre-MTP priority.

Deferred tuning levers (recorded in steering): slots x context shape envelope,
NCCL ring/topology pinning, C4 KV spill. These are situational and only matter
once a binding constraint beyond EP is identified; EP dominates today.
