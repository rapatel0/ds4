# Sprint 209 Deferred Items

Date: 2026-05-23

## Full TP8 Runtime Branch

- **What**: Implement a full TP8 runtime branch with request batching,
  sharded-KV attention, routed/shared FFN, output head, and serving integration.
- **Why deferred**: Sprint 209 first validates a bounded one-layer TP8 shape.
- **Target sprint**: Sprint 210+ if Sprint 209 gates pass.
- **Prerequisites**: Positive one-layer TP8 timing and correctness.
- **Files**: Future TP-only runtime files, not `ds4_v100_scheduler.*`.

## Full Sharded Attention

- **What**: Implement DS4 compressed attention over TP8-sharded KV, including
  ratio-4 indexer behavior and ratio-128 compressed row selection.
- **Why deferred**: Sprint 209 only proves shard ownership and a bounded
  layer-like compute boundary.
- **Target sprint**: Future TP attention sprint.
- **Prerequisites**: TP8 one-layer gate passes.
- **Files**: Future TP-only attention/KV files.

## TurboMind TP8 Routed FFN Pack Conversion

- **What**: Produce offline TP8-routed expert packs with split-axis metadata
  and TurboMind-compatible descriptors.
- **Why deferred**: Sprint 209 may use synthetic compute or a minimal
  TurboMind adapter, but full pack conversion should follow topology proof.
- **Target sprint**: Future pack sprint.
- **Prerequisites**: Positive one-layer TP8 prototype.
- **Files**: Future TP pack tools and manifests.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| Full TP8 runtime branch | Sprint 210+ | One-layer TP8 gate |
| Full sharded attention | Future | TP8 layer proof |
| TurboMind TP8 routed FFN pack conversion | Future | Selected TP8 compute body |
