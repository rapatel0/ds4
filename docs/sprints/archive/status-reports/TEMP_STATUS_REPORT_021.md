# TEMP Status Report 021

Date: 2026-05-24

## Current Topline

TP/EP serving is operational but not model-correct yet.

Current default 32-slot / 256K reference-vector run:

```text
mode:                  TP8/EP8 HTTP serving
context:               256K configured
slots:                 32
active microbatch:     32
reference vector:      short_reasoning_plain
expected text:         16
actual text:           01
generated token:       2616
wall tok/s:            46.408976
decode tok/s:          50.847122
result:                HTTP completed, parity failed
```

This is a correctness sprint, not a throughput sprint. The current decode
number should be treated as diagnostic because the path has extra semantic
instrumentation and parity gates enabled.

## Latest Change Under Test

The current local change adds route-local activation scaling behind the
reference-HC diagnostic gate:

```text
DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1
```

The purpose is to let the more reference-like HC reduce path feed the V100
TurboMind FP16 activation boundary without overflowing the routed gate/up
kernel. The scaling is intentionally not enabled for the stable default path.

Implementation shape:

- compute per-route max abs over the full hidden vector;
- scale route activations down before packing to FP16;
- store per-route inverse scale;
- restore the linear scale inside the routed gate/up SwiGLU clamp kernel;
- keep the default non-reference route pack path unchanged.

## Latest Cluster Results

Default regression run:

```text
cluster log:
  /workspace/logs/sprint309-route-scale-default/20260524-152526

result:
  completed HTTP parity harness
  expected: 16
  actual:   01
  decode:   50.847122 tok/s
```

Reference-HC route-scaling run:

```text
cluster log:
  /workspace/logs/sprint309-route-scale-reference-hc/20260524-152827

result:
  no longer fails immediately at layer 0
  route_input max_abs is scaled to 32
  routed gate/up/down remain finite through early layers
  fails later at layer 32 with decode_finite_bad=16384 and rc=5
```

This is progress compared with the prior reference-HC run, which failed much
earlier from activation overflow. The remaining failure now appears downstream
in accumulated layer semantics/state, not in the first routed activation pack.

Layer-window rerun:

```text
cluster log:
  /workspace/logs/sprint309-layer32-reference-hc-window/20260524-153920

result:
  route-local scaling works through the routed FFN path
  compose_next_hidden at layer 32 is still finite on every rank
  final_hc_shard at layer 32 has 2048 non-finite values per rank
```

Representative stats:

```text
layer 30 hc_current_full max_abs: 1.20720941e+15
layer 31 hc_current_full max_abs: 4.01145758e+15
layer 32 hc_current_full max_abs: 9.217978e+15

layer 32 compose_next_hidden: finite_bad=0
layer 32 final_hc_shard:     finite_bad=2048 per rank
```

Guarded diagnostic run:

```text
cluster log:
  /workspace/logs/sprint309-reference-hc-state-guard/20260524-154632

config:
  DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1
  DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD=1

result:
  completed HTTP parity harness
  expected: 16
  actual:   [$
  decode:   47.823187 tok/s
```

Default regression after adding the guard:

```text
cluster log:
  /workspace/logs/sprint309-default-after-state-guard/20260524-155108

result:
  completed HTTP parity harness
  expected: 16
  actual:   proiektuak
  decode:   44.905048 tok/s
```

## Current Diagnosis

The TP/EP implementation has crossed the infrastructure line:

- HTTP serving works.
- 32-slot / 256K admission works.
- resident dense cache, expert bindings, TP runtime, tokenizer, output head,
  session state, and per-step token feedback are wired.
- model-router EP routing is now active.
- true shared FFN is wired.

The blocker is now model semantic parity. The current TP/EP layer path still
has simplified attention/HC semantics and diagnostic state transitions that do
not yet match DS4 exactly.

The latest reference-HC experiment shows:

- default path remains stable;
- reference-HC route input can be made finite on V100;
- unguarded reference-HC first becomes non-finite in final-HC expansion at
  layer 32, not in routed FFN or compose;
- guarded reference-HC can execute all layers but still returns the wrong
  token, so the blocker is now semantic equivalence of the HC/attention bridge.

## Immediate Next Step

Next work:

1. compare `hc_split_rows_kernel` / `hc_expand_shard_kernel` against `ds4.c`
   and llama.cpp DeepSeek4 reference semantics;
2. replace the simplified HC/attention bridge with the real DS4
   compressed-KV/indexer update path;
3. keep `DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD` diagnostic-only.
