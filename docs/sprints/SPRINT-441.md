# Sprint 441: Masked-Copy HTTP Parity

## Objective

Validate full-shape masked compact-copy through operational HTTP response
parity and decide whether it should move toward promotion.

No PP/layer-split variants are in scope.

## Implementation

Fixed the HTTP validation helpers so they read DS4 metadata from both nested
and top-level diagnostic response formats:

```text
tools/ds4-v100-http-response-parity.py
tools/ds4-v100-http-readiness-check.py
```

This matters because selected-token responses carry `generated_token_sequence`,
`checksum`, `slots`, `ctx`, and KV flags at the top level.

## V100 Evidence

Chat A/B artifacts:

```text
/localpool/ds4/workspace/logs/sprint440-masked-copy-http-chat-ab
```

Shape:

```text
8 requests / 8 slots / 256K ctx / 2 generated tokens/request
```

Result:

```text
control ready: true
candidate ready: true
response parity: 8/8 matched

server generated decode tok/s:    14.079750 -> 14.047247
server continuation decode tok/s: 13.989955 -> 13.928785
client generated tok/s:            1.214761 ->  1.180515
avg GPU util:                       8.6957% ->  7.3958%
min free VRAM:                      4674 MiB both
```

Selected-token parser recheck:

```text
/localpool/ds4/workspace/logs/sprint440-masked-copy-http-ab
top-level response parity: true, 8/8 matched
top-level readiness recheck: true
```

## Decision

Keep masked compact-copy diagnostic-only.

The optimization is now token-safe in the HTTP chat A/B, but it does not improve
the operational serving topline. The remaining bottleneck is not this copy in
isolation.

## Next

Move to the real executor lever:

- static host-visible graph shape;
- device-side active-route masks/totals;
- no host route-count dependency;
- no token drift;
- avoid useful work for inactive fixed-capacity routed FFN rows.
