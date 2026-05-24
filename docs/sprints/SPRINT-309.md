---
sprint: 309
title: TP/EP Reference-HC Stability Boundary
status: in_progress
started: 2026-05-24
branch: claude-takeover
---

# Sprint 309 - TP/EP Reference-HC Stability Boundary

## Goal

Keep the TP/EP-only serving path moving toward model-correct DS4 output by
turning the Sprint 308 reference-HC overflow into a localized, reproducible
semantic boundary.

## Scope

This sprint does not re-open PP/layer-split work. It stays on the TP8/EP8
HTTP serving harness and focuses on the HC-current / final-HC bridge used by
the diagnostic DS4 layer path.

## Changes

- Added route-local activation scaling behind
  `--reference-hc-reduce-gate`.
  - Per-route max abs is computed over the full hidden vector.
  - Route input is scaled before FP16 packing into the TurboMind routed
    gate/up path.
  - The inverse scale is carried per route and applied before the routed
    SwiGLU clamp.
  - The default non-reference route pack path is unchanged.
- Added focused reference-HC tensor diagnostics for layers `30-32`.
  - `hc_current_shard`
  - `hc_current_full`
  - `hc_ffn_normed`
  - `route_inv_scale`
  - routed `route_input`, `route_gate_up`, `route_gated`, `route_down`
  - shared FFN gate/up/mid/down
  - `compose_next_hidden`
  - `final_hc_shard`
- Added a separate opt-in diagnostic state guard:
  - CLI: `--reference-hc-state-guard-gate`
  - launcher env: `DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD=1`
  - This implies `DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1`.
  - It clamps HC state after DS4-style final-HC expansion to keep the
    diagnostic bridge finite.
  - It is not a production correctness path.

## V100 Evidence

### Default Regression

```text
log:
  logs/from-cluster/sprint309-default-after-state-guard/20260524-155108

config:
  32 slots / 256K
  model-router routes on
  routed FFN norm input on
  true shared FFN on
  reference HC reduce off
  reference HC state guard off

result:
  HTTP completed
  expected text: 16
  actual text:   proiektuak
  token:         118235
  wall tok/s:    41.265575
  decode tok/s:  44.905048
```

This remains a parity failure, but it is an ordinary model-output mismatch,
not a serving failure.

### Unguarded Reference-HC Reduce

```text
log:
  logs/from-cluster/sprint309-layer32-reference-hc-window/20260524-153920

result:
  HTTP 500 from tp_ep_decode_failed
  failed layer: 32
  compose_next_hidden layer 32: finite on all ranks
  final_hc_shard layer 32: 2048 non-finite values per rank
```

Key tensor stats:

```text
layer 30 hc_current_full max_abs: 1.20720941e+15
layer 31 hc_current_full max_abs: 4.01145758e+15
layer 32 hc_current_full max_abs: 9.217978e+15

layer 32 compose_next_hidden:
  finite_bad=0 on all ranks
  max_abs about 6.96e15 - 9.22e15

layer 32 final_hc_shard:
  finite_bad=2048 on every rank
  first_bad=0 on every rank
```

This localizes the runtime instability to DS4-style final-HC expansion after
the compose output, not to TurboMind route packing or route down output.

### Guarded Reference-HC Diagnostic

```text
log:
  logs/from-cluster/sprint309-reference-hc-state-guard/20260524-154632

config:
  reference HC reduce on
  reference HC state guard on

result:
  HTTP completed
  expected text: 16
  actual text:   [$
  token:         38218
  wall tok/s:    43.761738
  decode tok/s:  47.823187
```

The guard lets the diagnostic path complete all layers again. It does not
prove correctness; it proves the remaining failure is semantic mismatch once
the known HC-state blow-up is contained.

## Interpretation

Sprint 309 rules out the earlier suspicion that the current reference-HC
failure is still primarily a TurboMind routed activation overflow. Route-local
scaling keeps the routed path finite. The first hard instability is now the
HC recurrent state update itself: the simplified bridge produces enormous HC
state values by layer 30, and final-HC expansion creates non-finite rows at
layer 32.

The next correctness work should focus on the DS4 HC/attention semantics:

- compare the current `hc_split_rows_kernel` and `hc_expand_shard_kernel`
  bridge against the reference `ds4.c` and llama.cpp DeepSeek4 memory path;
- verify whether `hc_current` should be derived from the same rows and
  coefficients currently used here;
- replace the simplified attention body with the real compressed-KV/indexer
  update path, or prove the bridge can be made semantically equivalent;
- keep `DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD` diagnostic-only.

## Definition of Done

- V100 build passes. Complete.
- Default TP/EP HTTP path still starts and returns a parity result. Complete.
- Unguarded reference-HC failure is localized to a concrete layer/tensor
  boundary. Complete.
- Guarded reference-HC diagnostic completes HTTP parity without runtime
  failure. Complete.
- Evidence is copied into `logs/from-cluster`. Complete.

