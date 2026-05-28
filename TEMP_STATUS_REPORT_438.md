# TEMP Status Report 438

## Focus

TP/EP only. Implemented the first full-shape internal route-copy mask.

## Added

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` now accepts:

```text
--post-attention-masked-compact-copy-gate
```

This keeps the host-visible graph copy shape fixed and masks active/inactive
route rows inside the copy kernel.

## V100 Results

```text
full cap baseline: 38.765556 tok/s, checksum=6775636869
masked copy:       47.153014 tok/s, checksum=6304609080
masked forced:     54.037323 tok/s, checksum=53235842
```

The output-head diagnostic unexpectedly did not emit for the masked-copy runs,
including the forced `--diagnostic-output-head` run.

## Decision

Keep masked copy diagnostic-only. It is the right shape of optimization and
shows positive proxy speed, but it is not validated.

## Next

Run token-level parity through HTTP or fix the all-layer output-head harness for
this candidate before doing any further performance interpretation.

No GPU jobs were left running after the tests.
