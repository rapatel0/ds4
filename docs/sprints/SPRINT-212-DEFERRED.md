# Sprint 212 Deferred Items

Date: 2026-05-23

## TP4/PP1 Runtime Branch

- **What**: Build a separate TP4/PP1 runtime branch with layer ownership,
  request batching, sharded KV ownership, routed/shared FFN, and output head.
- **Why deferred**: Sprint 212 first validates the low-bit TP4 layer body and
  reduction boundary.
- **Suggested sprint**: Sprint 213 if Sprint 212 passes.
- **Files**: Future TP-only runtime files, not `ds4_v100_scheduler.*`.

## TP8 Shard-256 Kernel Design

- **What**: Design or adapt MXFP4 kernels that are numerically valid and
  performant for TP8 `mid_shard=256`.
- **Why deferred**: Sprint 211 rejected the current generic TurboMind shape;
  Sprint 212 pivots to the known-correct TP4 path first.
- **Suggested sprint**: Future kernel sprint if TP4 is insufficient.
- **Files**: Future TurboMind/kernel files.

## Sharded Attention/KV

- **What**: Implement TP-aware DS4 attention over sharded compressed KV.
- **Why deferred**: The low-bit routed FFN topology must first be selected.
- **Suggested sprint**: Future TP attention sprint.
- **Files**: Future TP-only attention/KV files.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| TP4/PP1 runtime branch | Critical for TP4 serving | Sprint 213 if gate passes | Future TP-only runtime files |
| TP8 shard-256 kernel design | Important | Future kernel sprint | Future TurboMind/kernel files |
| Sharded attention/KV | Important | Future | Future TP-only attention/KV files |
