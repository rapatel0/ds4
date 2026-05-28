# TEMP Status Report 434

## Focus

TP/EP only. Tested a graph-safe-looking static route cap as a shortcut for
reducing fixed-capacity routed FFN work.

## Added

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` now accepts:

```text
--post-attention-static-rank-route-cap N
```

This is diagnostic and default-off. It keeps full route-plan buffers but uses a
smaller fixed per-rank launch row count and prints device route-total overflow
audits.

## V100 Results

All runs were `8 slots / 256K / persistent graph replay`.

```text
full cap: 39.491776 tok/s, aggregate_routes=384, checksum=3211778491
cap 32:   44.163120 tok/s, aggregate_routes=256, checksum=1709346105
cap 16:   50.502275 tok/s, aggregate_routes=128, checksum=6493007747
```

Overflow:

```text
cap 32: none
cap 16: none
```

Output-head follow-up:

```text
full cap: first_token=50845, first_logit=18.253084183, 36.846896 tok/s
cap 16:   first_token=106720, first_logit=17.242784500, 50.408429 tok/s
```

## Interpretation

The cap reduces graph replay time, but it is not correctness-preserving:
the selected output token changes despite zero route overflow. Checksums also
change, but full-cap graph checksums are not stable enough to use as the sole
criterion. Token-level output-head parity is the decisive failure.

## Decision

Reject static caps as a serving strategy. Keep the gate only as a diagnostic.

The next implementation must keep `total_tokens = route_capacity` for graph
capture and move the inactive-row skip inside the TurboMind DS4 executor or a
new dedicated routed FFN kernel.

No GPU jobs were left running after the tests.
