# TEMP Status Report 431

Date: 2026-05-27

## Focus

TP/EP only. Investigated whether removing fixed-capacity routed FFN execution
is the next material lever.

## Result

The final harness builds cleanly on the V100 node. I removed the attempted
host route-count oracle because it is not correctness-preserving.

Directional oracle measurement:

```text
log: /localpool/ds4/workspace/logs/sprint431-host-route-count-oracle/alllayers-oracle-slots8-clean.stdout
shape: 8 slots, 256K ctx, 43 layers, 1 decode step
route audit: 2064 checked, 0 missing, 0 mismatch, 0 invalid
projected decode: 44.270973 tok/s
```

Comparison:

```text
Sprint 429 fixed capacity: 34.738433 tok/s
Sprint 430 gated packer:  34.571189 tok/s
Sprint 431 oracle:        44.270973 tok/s
```

## Interpretation

The performance direction is real: reducing routed execution from
`aggregate_routes=384` to the actual `48` rows at 8 slots materially improves
the direct graph probe.

But the host oracle is invalid for production because route totals are produced
inside the captured post-attention path and can differ from warmup. The clean
log shows `routes=6` per rank while the captured route audit has imbalanced
rank totals, so a host-seeded launch can under-execute routed rows.

## Next

Build a graph-safe actual-route executor at the TurboMind boundary. It must use
device route totals, masks, or device-side compaction; it cannot depend on host
route-count reads.
