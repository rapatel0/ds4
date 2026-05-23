# Sprint 211 Deferred Items

Date: 2026-05-23

## TP8 Runtime Ownership

- **What**: Build request batching, layer ownership, scratch, KV, and output
  contracts for a TP8 runtime branch.
- **Why deferred**: Sprint 211 is still a low-bit expert-body gate, not
  serving integration.
- **Suggested sprint**: Sprint 212+ if low-bit TP8 expert evidence passes.
- **Files**: Future TP-only runtime files, not `ds4_v100_scheduler.*`.

## Production TP8 Collective

- **What**: Replace simple peer-copy reduction with NCCL-grade or fused
  reduce/scatter primitives.
- **Why deferred**: Sprint 211 first checks whether MXFP4 expert compute is
  worth reducing at all.
- **Suggested sprint**: After the low-bit body gate.
- **Files**: Future TP-only collective files.

## Sharded Attention/KV

- **What**: Implement DS4 ratio-4 / ratio-128 compressed attention over
  TP-sharded KV.
- **Why deferred**: Sprint 211 focuses on routed FFN precision/layout fidelity.
- **Suggested sprint**: Future TP attention sprint.
- **Files**: Future TP-only attention/KV files.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| TP8 runtime ownership | Critical for final vision | Sprint 212+ | Future TP-only runtime files |
| Production TP8 collective | Important | After low-bit body gate | Future TP-only collective files |
| Sharded attention/KV | Important | Future | Future TP-only attention/KV files |
