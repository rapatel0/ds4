# Sprint 422: Rank-Major Attention Projection Input

## Objective

Move the positive rank-local attention projection gate toward the stricter
rank-major TP/EP strategy: consume NCCL allgather rank-major hidden state
directly and avoid normalizing a slot-major full tensor on each rank.

No PP/layer-split work is in scope.

## Change

The existing gate:

```text
--true-ds4-attention-projection-rank-local-input-gate
```

now uses rank-major input when HC-current NCCL allgather is active. The new
kernel:

```text
fill_two_hidden_inputs_half_from_rank_major_norm_kernel
```

fuses RMS norm, `attn_norm.weight`, rank-major addressing, F32-to-F16
conversion, and two projection input fills.

Fallback remains the previous slot-major rank-local path when rank-major input
is unavailable.

## Evidence

Resident layer 2:

```text
/localpool/ds4/workspace/logs/sprint422-rankmajor-attn-proj/resident-layer2-rankmajor-clean/
```

All-layer direct:

```text
/localpool/ds4/workspace/logs/sprint422-rankmajor-attn-proj/full-rankmajor-slot8-tokens4-scratch256/
```

## Results

| Mode | Generated decode tok/s | Continuation decode tok/s | Checksum |
|---|---:|---:|---:|
| Sprint 416 baseline | 84.072506 | 94.326524 | 4335215310 |
| Sprint 416 rank-local slot-major | 92.702737 | 105.428529 | 4335215310 |
| Sprint 422 rank-major fused | 93.586972 | 106.584476 | 4335215310 |

Resident layer 2 graph nodes dropped from `789` to `773`, with checksum
unchanged.

## Decision

Continue rank-major conversion. The next target should be FFN/router RMS norm
and route input packing, because those still depend on materialized full hidden
state and device-0-oriented control flow.
