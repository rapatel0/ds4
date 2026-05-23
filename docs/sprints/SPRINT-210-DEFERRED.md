# Sprint 210 Deferred Items

Date: 2026-05-23

## Low-Bit TurboMind TP8 Expert Body

- **What**: Replace the FP16 fixture GEMM body with a TP8 low-bit expert body
  using copied/adapted TurboMind MXFP4/FP8 kernels and TP-aware pack metadata.
- **Why deferred**: Sprint 210 first proves real resident TP8 tensor-core
  layer shape and reduction timing before adding pack conversion and low-bit
  descriptors.
- **Suggested sprint**: Sprint 211 if Sprint 210 passes.
- **Files**: Future TP-only pack/runtime files and TurboMind adapter changes.

## Sharded Attention/KV Execution

- **What**: Implement ratio-4 / ratio-128 compressed KV row selection and
  attention over TP8-owned KV shards.
- **Why deferred**: Sprint 210 focuses on FFN-shaped useful compute inside the
  TP boundary.
- **Suggested sprint**: Sprint 211+ depending on whether FFN or attention is
  the larger measured gap.
- **Files**: Future TP-only attention/KV files.

## Full TP8 Serving Runtime

- **What**: Request batching, decode loop, KV paging, output head, MTP, and
  API-serving integration for TP8.
- **Why deferred**: Requires positive real-layer TP8 evidence first.
- **Suggested sprint**: Future, after low-bit FFN and attention gates.
- **Files**: Future TP-only runtime files, not `ds4_v100_scheduler.*`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Low-bit TurboMind TP8 expert body | Important | Sprint 211 | Future TP-only pack/runtime files |
| Sharded attention/KV execution | Important | Sprint 211+ | Future TP-only attention/KV files |
| Full TP8 serving runtime | Critical for final vision | Future | Future TP-only runtime files |
