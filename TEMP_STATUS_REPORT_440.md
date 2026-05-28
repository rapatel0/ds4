# TEMP Status Report 440

## Focus

TP/EP only. Tested whether the post-attention FFN input path can skip the
device-0 slot-major full-hidden gather/RMSNorm once rank-major shared input,
rank-major route input, and rank-major router logits are enabled.

## Added

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` now reports per-layer:

```text
slot_major_ffn_norm
```

and has explicit diagnostic controls:

```text
--post-attention-slot-major-ffn-norm-gate
--post-attention-skip-slot-major-ffn-norm-gate
```

The safe default is now to keep the slot-major FFN norm path unless the skip
gate is explicitly set.

## V100 Results

Same-binary 4-slot / 256K / all-layer / persistent graph replay A/B:

```text
control slot-major norm:
  log: /localpool/ds4/workspace/logs/sprint433-rank-major-slotnorm-skip/host-ab/control-slotmajor-bg.stdout
  projected_slot_step_tok_s: 23.667788
  first_token: 45178
  graph nodes: 51810
  slot_major_ffn_norm layers: 43/43

rank-major skip diagnostic:
  log: /localpool/ds4/workspace/logs/sprint433-rank-major-slotnorm-skip/host-ab/candidate-rankmajor-fg.stdout
  projected_slot_step_tok_s: 23.951210
  first_token: 50845
  graph nodes: 51423
  slot_major_ffn_norm layers: 0/43
```

## Decision

Do not promote.

The skip path is graph-capturable and removes 387 graph nodes, but selected
token parity fails. The throughput delta is also small: about +1.2% decode
tok/s at this shape.

This means the device-0 slot-major FFN norm is still semantically coupled to a
downstream consumer or state path. It is a bad topology, but not safe to remove
until that dependency is eliminated or replaced with a rank-major equivalent.

## Next

Keep pushing rank-major, but target the measured larger bottleneck:

- graph-safe full-shape route masking / executor work;
- token-level HTTP parity for masked-copy candidates;
- then a true full-shape routed FFN executor that consumes device route masks
  internally while keeping host-visible graph dimensions static.

No GPU jobs were left running after the run.
