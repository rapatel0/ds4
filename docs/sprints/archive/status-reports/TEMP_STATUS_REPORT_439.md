# TEMP Status Report 439

## Focus

TP/EP only. Fixed the masked-copy parser bug and reran output-head validation.

## Fixed

`--post-attention-masked-compact-copy-gate` no longer skips the next CLI flag.
This was why Sprint 438 did not emit output-head diagnostics.

## V100 Result

Same rebuilt harness:

```text
masked copy:
  54.281002 tok/s
  first_token=50845
  first_logit=20.302221298

full cap repeat:
  38.706401 tok/s
  first_token=164
  first_logit=19.380706787
```

## Decision

Masked copy remains diagnostic-only. It is faster, but token parity is not
proven, and the full-cap smoke output-head result is unstable across repeats.

## Next

Use HTTP response parity rather than single all-layer smoke output-head parity
for this candidate.

No GPU jobs were left running after the tests.
