# TEMP Status Report 441

## Focus

TP/EP only. Validated the full-shape masked compact-copy route movement through
HTTP response parity instead of relying on all-layer smoke output-head probes.

## Harness Fix

`tools/ds4-v100-http-response-parity.py` and
`tools/ds4-v100-http-readiness-check.py` now accept DS4 diagnostic metadata from
either:

```text
body["ds4_v100"]
```

or the response body itself. The selected-token endpoint emits fields such as
`generated_token_sequence`, `checksum`, `slots`, and `ctx` at the top level, so
the old parsers reported vacuous parity for that endpoint.

## V100 Results

Chat HTTP A/B:

```text
artifact: /localpool/ds4/workspace/logs/sprint440-masked-copy-http-chat-ab
shape: 8 requests / 8 slots / 256K ctx / 2 generated tokens/request
control ready: true
candidate ready: true
response parity: 8/8 matched
```

Topline:

```text
server generated decode tok/s:      14.079750 -> 14.047247
server continuation decode tok/s:   13.989955 -> 13.928785
client generated tok/s:              1.214761 ->  1.180515
avg GPU util:                         8.6957% ->  7.3958%
min free VRAM:                        4674 MiB both
```

Selected-token parser recheck after the harness fix:

```text
artifact: /localpool/ds4/workspace/logs/sprint440-masked-copy-http-ab
top-level response parity: true, 8/8 matched
top-level readiness recheck: true
```

## Decision

Do not promote masked compact-copy.

It preserves chat response parity, but the operational HTTP result is flat to
slightly slower at this shape. The selected-token result had a small positive
client/proxy signal, but chat serving is the stronger promotion gate.

## Next

Stop spending promotion effort on masked compact-copy by itself. The remaining
large lever is a full-shape routed FFN executor that keeps graph-visible launch
dimensions static while consuming route masks/totals internally so inactive
fixed-capacity rows do not do useful TurboMind/CUTLASS work.

No GPU jobs were intentionally left running by this sprint.
