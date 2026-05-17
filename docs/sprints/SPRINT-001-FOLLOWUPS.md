# SPRINT-001 Follow-Up Items

Follow-ups discovered during Sprint 001 execution.

## 1. Source Loader Name And Type Delta

**What:** Update the DS4 loader/type table for the measured source model:
`MXFP4`, `F8_E4M3_B128`, BF16 output/embedding, `attn_kv_latent.weight`,
`attn_compress_*`, `indexer.compress_*`, and `hc_head_*`.

**Why:** The Sprint 001 inventory showed the high-intelligence source GGUF does
not use the older q2/q4 DS4 tensor names or output dtype assumptions.

**Severity:** Critical.

**Suggested sprint:** Sprint 002.

**Files:** `ds4.c`, `tools/ds4-v100-plan.c`, future packer files.

## 2. Inventory-Backed Planner Input

**What:** Teach `tools/ds4-v100-plan` to consume the inventory TSV or future
pack manifest instead of relying only on static constants.

**Why:** Static planning was sufficient for Sprint 001, but implementation
should close the loop between exact model inventory and per-GPU pack bytes.

**Severity:** Important.

**Suggested sprint:** Sprint 002 or Sprint 003.

**Files:** `tools/ds4-v100-plan.c`, future manifest parser.

## 3. Local Test Model Availability

**What:** Provide a documented local test-model path or skip mode for
`make test` when `ds4flash.gguf` is intentionally absent.

**Why:** `make test` compiled successfully but failed at runtime because the
local model symlink was missing.

**Severity:** Nice-to-have.

**Suggested sprint:** When local test automation is hardened.

**Files:** `Makefile`, `tests/ds4_test.c`, README or contributor docs.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Source loader name and type delta | Critical | Sprint 002 | `ds4.c`, planner/packer files |
| Inventory-backed planner input | Important | Sprint 002 or 003 | `tools/ds4-v100-plan.c`, manifest parser |
| Local test model availability | Nice-to-have | Test hardening | `Makefile`, `tests/ds4_test.c`, docs |
