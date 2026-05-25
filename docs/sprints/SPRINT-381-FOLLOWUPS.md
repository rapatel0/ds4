# Sprint 381 Follow-Ups

## Longer E5M2 KV Parity Gate

- **What**: Run repeated longer selected-token and chat A/B cases for
  `--fp8-e5m2-kv-gate`, including generated token checksum/sequence parity
  beyond the short 4-token run.
- **Why**: Sprint 381 showed E5M2 is row-correct and faster in short direct and
  HTTP selected-token runs, but E5M2 has lower mantissa precision than E4M3.
  It should not become the default until longer continuation quality/parity is
  proven.
- **Severity**: Important before promotion.
- **Suggested sprint**: Dedicated parity/soak sprint if E5M2 remains a
  candidate.
- **Files**:
  `ds4_v100_tp_runtime.cu`,
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`,
  `tools/ds4-v100-tp-ep-profile.py`.

## 32-Slot / 256K VRAM Admission Margin

- **What**: Add a serving admission/report check that records free VRAM margin
  after dense cache, TP runtime, and scratch allocation at the default
  `32` slot / `256K` shape.
- **Why**: Sprint 381's control and E5M2 HTTP runs both reported `32418 MiB`
  max used on a `32 GB` V100, and one immediate E5M2 HTTP attempt failed with
  CUDA OOM before readiness. The runtime is operating with tens of MiB of slack,
  which is too tight for production.
- **Severity**: High for production readiness.
- **Suggested sprint**: Next memory-admission sprint or fold into the next
  serving hardening sprint.
- **Files**:
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`,
  `tools/ds4-v100-run-appliance.sh`,
  `tools/ds4-v100-tp-ep-profile.py`.

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Longer E5M2 KV parity gate | Important | Dedicated E5M2 parity/soak sprint | `ds4_v100_tp_runtime.cu`, `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, `tools/ds4-v100-tp-ep-profile.py` |
| 32-slot / 256K VRAM admission margin | High | Next memory-admission or serving hardening sprint | `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, `tools/ds4-v100-run-appliance.sh`, `tools/ds4-v100-tp-ep-profile.py` |
