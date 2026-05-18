# SPRINT-014 Deferred Items

These items are intentionally outside Sprint 014.

## Real Layer Compute

**What:** Consume validated descriptors to run real layer compute through
attention, shared/routed experts, and output-head selected-token comparison.

**Why deferred:** Sprint 014 first creates the fail-closed descriptor contract
needed before compute can safely bind real pack bytes.

**Target sprint:** Sprint 015.

## Multi-Layer And Final-Layer Descriptor Coverage

**What:** Extend descriptor validation to ratio-128 layers, final layer 42, and
all 43 layers.

**Why deferred:** Layer 2 covers the largest surface first: ratio-4,
compressor/indexer, router, routed experts, shared experts, and HC controls.

**Target sprint:** Sprint 015+.

## Public Deployment And Throughput

**What:** CLI/server deployment, throughput tuning, MTP, and tensor parallelism.

**Why deferred:** Serving and optimization should follow real descriptor-bound
selected-token correctness.

**Target sprint:** Sprint 016+.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Real layer compute | Sprint 015 | Needs descriptor gate |
| Multi-layer descriptor coverage | Sprint 015+ | Needs layer 2 descriptor contract |
| Deployment/throughput/MTP | Sprint 016+ | Needs descriptor-bound correctness |
