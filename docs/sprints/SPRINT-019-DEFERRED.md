# SPRINT-019 Deferred Items

## Full 43-Layer Selected-Token Decode

Sprint 019 targets one representative ratio-4 layer. Full 43-layer selected
token correctness remains deferred until the single-layer executor can be
walked across SWA-only, ratio-4, and ratio-128 layers.

## Public Serving

Server exposure remains deferred until the appliance can produce selected-token
evidence from the real layer-scheduled path.

## MTP

MTP speculative decoding remains deferred. It should be enabled only after the
base layer-scheduled decode path is correct and measurable.

## Multi-Slot Throughput Optimization

Slot batching and wavefront scheduling remain deferred until the single-slot
runtime path is correct on V100 hardware.

## F8 KV And INT8 Runtime Variants

F16 KV and source-FP8/MXFP4 packs remain the Sprint 019 baseline. F8 KV and
INT8 runtime variants are deferred until the baseline layer output is correct
and has hardware measurements.

## Tensor Parallel Execution

Tensor-parallel variants described in
`docs/architecture/DS4-V100-LAYOUT.md` remain evaluation candidates. Sprint 019
does not implement them.

