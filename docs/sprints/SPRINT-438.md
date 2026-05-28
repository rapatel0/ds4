# Sprint 438: Full-Shape Masked Compact Copy

## Objective

Preserve the graph-visible compact route copy shape while avoiding useful
remote reads for inactive route rows.

No PP/layer-split variants are in scope.

## Implementation

Added a default-off diagnostic:

```text
--post-attention-masked-compact-copy-gate
```

The gate keeps:

- `r.routes = route_capacity`;
- compact segment length fixed;
- graph launch dimensions fixed;
- TurboMind `total_tokens` fixed.

It replaces the graph compact route copy kernel with a route-total-aware kernel
that copies active route rows and writes zero for inactive route rows.

## V100 Evidence

Logs:

```text
/localpool/ds4/workspace/logs/sprint438-masked-copy/maskedcopy-slots8-graph-head.stdout
/localpool/ds4/workspace/logs/sprint438-masked-copy/maskedcopy-slots8-graph-head-forced.stdout
```

Shape:

```text
8 slots / 256K / 43 layers / persistent graph replay
```

Results:

```text
full cap baseline: projected_slot_step_tok_s=38.765556, checksum=6775636869
masked copy:       projected_slot_step_tok_s=47.153014, checksum=6304609080
masked forced:     projected_slot_step_tok_s=54.037323, checksum=53235842
```

## Decision

Do not promote yet.

The performance direction is positive and the host-visible route envelope stays
full, but the output-head diagnostic did not emit in the masked-copy runs even
when `--diagnostic-output-head` was forced. The scaffold checksum also changed,
so the candidate needs token-level validation before it can be trusted.

## Next

Fix or bypass the output-head validation gap:

- run an HTTP parity A/B with masked copy at the smallest admitted shape; or
- fix the all-layer smoke harness so output-head diagnostics always emit with
  masked copy; then
- only if selected tokens match, repeat at 8 and 16 slots and inspect Nsight
  for reduced remote-read traffic.
