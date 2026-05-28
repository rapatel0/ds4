# Sprint 440: Rank-Major FFN Norm Skip Diagnostic

## Objective

Test whether the TP/EP post-attention FFN input path can stop materializing the
full hidden tensor on device 0 for FFN RMSNorm once the surrounding consumers
are rank-major.

No PP/layer-split variants are in scope.

## Implementation

Added per-layer visibility for:

```text
slot_major_ffn_norm
```

and added explicit diagnostic gates:

```text
--post-attention-slot-major-ffn-norm-gate
--post-attention-skip-slot-major-ffn-norm-gate
```

The skip is intentionally explicit. The safe default keeps the legacy
slot-major FFN norm path because this sprint found a selected-token mismatch.

## V100 Evidence

Logs:

```text
/localpool/ds4/workspace/logs/sprint433-rank-major-slotnorm-skip/host-ab/control-slotmajor-bg.stdout
/localpool/ds4/workspace/logs/sprint433-rank-major-slotnorm-skip/host-ab/candidate-rankmajor-fg.stdout
```

Shape:

```text
4 slots / 256K / 43 layers / persistent graph replay / diagnostic output head
```

Results:

```text
control:
  projected_slot_step_tok_s=23.667788
  first_token=45178
  capture_nodes=51810
  slot_major_ffn_norm_layers=43

rank-major skip:
  projected_slot_step_tok_s=23.951210
  first_token=50845
  capture_nodes=51423
  slot_major_ffn_norm_layers=0
```

## Decision

Reject promotion.

The graph-captured skip path runs and removes some graph work, but the selected
token changed. The performance improvement is only about +1.2%, so this is not
worth pursuing before the larger fixed-capacity route/executor bottleneck.

## Next

Treat slot-major FFN norm removal as a later cleanup after the true downstream
rank-major dependency is identified. The main active line remains:

- full-shape masked route movement with token-level parity;
- full-shape routed FFN executor with internal device-side active-route masks;
- persistent graph replay with static host-visible launch dimensions.
