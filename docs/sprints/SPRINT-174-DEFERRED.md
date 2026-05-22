# Sprint 174 Deferred Items

Date: 2026-05-22

## 8-Way TP/EP Production Topology

- **What**: Extend routed-FFN TP/EP from one owner/peer pair to all eight V100s
  with production scheduling.
- **Why deferred**: Sprint 174 first needs a bounded persistent boundary that
  proves correctness and performance on one NVLink pair.
- **Target sprint**: Future, after Sprint 174 served A/B.
- **Prerequisites**: Positive persistent two-GPU TP/EP result.
- **Files**: `ds4_cuda.cu`, `ds4_v100_scheduler.*`, `ds4_v100_layer_execute.*`.

## Monolithic Routed-FFN Kernel

- **What**: Fuse gate/up, activation, down, and route reduce into one larger
  kernel that removes `mid_half` and `down_routes`.
- **Why deferred**: Sprint 174 prioritizes TP/EP because Sprint 173's local
  `a_half` removal was correct but slower.
- **Target sprint**: Future or immediate pivot if TP/EP fails.
- **Prerequisites**: Sprint 174 copy/compute timing decision.
- **Files**: `ds4_cuda.cu`, TurboMind probe/API files.

## Attention/Shared-FFN Parallelism

- **What**: Apply TP or fusion to attention and shared FFN paths.
- **Why deferred**: Routed FFN is the current bounded target with existing TP
  split evidence.
- **Target sprint**: Future.
- **Prerequisites**: Routed TP/EP decision.
- **Files**: `ds4_cuda.cu`, `ds4_v100_layer_execute.*`.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| 8-way TP/EP production topology | Future | Positive two-GPU persistent boundary |
| Monolithic routed-FFN kernel | Future/pivot | Sprint 174 TP/EP decision |
| Attention/shared-FFN parallelism | Future | Routed TP/EP decision |
