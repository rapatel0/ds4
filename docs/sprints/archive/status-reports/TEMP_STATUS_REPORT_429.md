# TEMP Status Report 429

Date: 2026-05-27

## Focus

TP/EP only. Implemented the first graph-safe post-attention route planner.

## Added

```text
--post-attention-fixed-capacity-route-plan-gate
```

The gate recomputes post-attention router selection inside persistent graph
capture/replay and updates route slots, route weights, offsets, and compact
route metadata on device without host route-count reads.

## V100 Results

Build passed on `gpu-01` with `sm_70`.

4-slot / 256K:

```text
log: /localpool/ds4/workspace/logs/sprint429-post-attn-fixed-route/alllayers-fixed-slots4.stdout
43/43 layers PASS
route audit: 1032 checked, 0 missing, 0 weight mismatch, 0 invalid
projected decode: 24.292901 tok/s
```

8-slot / 256K:

```text
log: /localpool/ds4/workspace/logs/sprint429-post-attn-fixed-route/alllayers-fixed-slots8.stdout
43/43 layers PASS
route audit: 2064 checked, 0 missing, 0 weight mismatch, 0 invalid
projected decode: 34.738433 tok/s
```

## Interpretation

Sprint 428 proved stale route metadata. Sprint 429 proves the post-attention
route plan can be recomputed and consumed inside CUDA graph replay without
route mismatch.

This is correctness-first and slower because fixed capacity forces every rank
to process `slots * top_k` routes. At 8 slots that is `384` aggregate rank
routes versus `48` actual routed entries.

## Current Blockers

- Need graph-safe actual-route execution, not fixed-capacity over-compute.
- Shared contiguous expert residency OOMs before graph capture; local
  per-layer expert bindings were used for these probes.
- Rank-major input should remain default-off until same-binary parity and a
  non-overcomputing graph route planner are proven.
