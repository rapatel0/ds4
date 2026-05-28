# TEMP Status Report 437

## Focus

TP/EP only. Tested compose-only static route capping after executor-only capping
failed.

## Added

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` now accepts:

```text
--post-attention-static-compose-route-cap N
```

This is diagnostic and default-off.

## V100 Result

Same shape:

```text
8 slots / 256K / all layers / persistent graph / output-head diagnostic
```

Result:

```text
full cap:       38.765556 tok/s, first_token=50845
compose cap 16: 49.381819 tok/s, first_token=164
```

The cap audit was overflow-free through the final layers, so this is not a
simple missing-route overflow.

## Decision

Reject compose-only static caps. They recover the speed direction but change
the output token.

## Current Bottleneck Read

The useful work-reduction lever is the route transfer/compose envelope, not the
TurboMind gate/down executor alone. However, reducing host-visible graph shape
or compact segment copy length is not correctness-safe in the current runtime.

## Next

Move route masking inside full-shape device code:

- preserve graph-visible launch/copy shapes;
- use device route counts/indices internally;
- skip or zero inactive route rows without changing visible segment shape;
- validate output-head token parity first, then benchmark.

No GPU jobs were left running after the test.
