# TEMP Status Report 435

## Focus

TP/EP only. I reran the static route-cap candidate with the lazy output-head
diagnostic so the decision is based on selected-token parity.

## Result

Shape:

```text
8 slots / 256K / 43 layers / persistent graph replay
```

Evidence:

```text
full cap: 36.846896 tok/s, first_token=50845, first_logit=18.253084183
cap 16:   50.408429 tok/s, first_token=106720, first_logit=17.242784500
```

The cap16 route audit had no overflow through the final layer:

```text
layer 42: cap=16, max_actual=9, overflow_ranks=0, PASS
```

## Decision

Static route caps are rejected for serving. They improve the graph proxy but
change the actual output token.

## Next

The next implementation target is a true static-envelope actual-route routed
FFN executor:

- preserve host-visible `total_tokens = route_capacity` for graph capture;
- pass device route totals or masks into the executor;
- skip inactive fixed-capacity rows inside TurboMind/CUTLASS or a dedicated
  DS4 routed FFN kernel;
- validate selected-token parity before considering throughput.

No GPU jobs were left running after the parity run.
