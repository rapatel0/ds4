# Sprint 380 Follow-Ups

## Fused TP4 Reduction/Compose Boundary

- **What**: Prototype a TP4 expert-output reduction that fuses partial hidden
  reduction with the existing EP compose path instead of copying every TP
  participant's full `[routes, 4096]` output back to root and reducing there.
- **Why**: Sprint 380 showed TP4 MXFP4 expert compute is correct and materially
  faster than the full reference, but the simple reduction dominates at larger
  route tiers. Total speedup was only `1.055x`, `0.891x`, and `0.927x` for
  `96`, `192`, and `384` routes respectively.
- **Severity**: Important if TP-sharded experts are revisited; otherwise
  defer.
- **Suggested sprint**: Dedicated TP4 reduction sprint before any serving
  integration attempt.
- **Files**:
  `tools/ds4-v100-tp4-turbomind-layer-smoke.cu`,
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`,
  `tools/ds4-v100-tp-experts-ab.py`.

## TP8 MXFP4 Shard-256 Kernel Shape

- **What**: Investigate or replace the TurboMind MXFP4 `mid_shard=256` expert
  shape if TP8 experts are ever reconsidered.
- **Why**: Sprint 380 reconfirmed TP8 correctness failure at `96`, `192`, and
  `384` routes, with large NaN counts. The current TP8 path is not a serving
  candidate.
- **Severity**: Nice-to-have. TP4 is the only plausible TP expert branch today.
- **Suggested sprint**: Only after TP4 reduction is proven useful, or if a new
  TurboMind kernel shape becomes available.
- **Files**:
  `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu`,
  `kernels/turbomind/`.

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Fused TP4 reduction/compose boundary | Important | Dedicated TP4 reduction sprint | `tools/ds4-v100-tp4-turbomind-layer-smoke.cu`, `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, `tools/ds4-v100-tp-experts-ab.py` |
| TP8 MXFP4 shard-256 kernel shape | Nice-to-have | After TP4 reduction or new kernel availability | `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu`, `kernels/turbomind/` |
