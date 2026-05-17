# SPRINT-002 Follow-Ups

## 1. Emit Per-GPU Shard Files

**What:** Add a packer that consumes
`docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` and writes immutable
per-GPU weight arenas.

**Why:** The manifest now records source offsets, owner GPU, source dtype, and
kernel family. The runtime still needs actual shard files or direct upload
plans derived from that contract.

**Severity:** Critical

**Suggested sprint:** Sprint 003

**Files:** `tools/`, future packer/runtime files

## 2. Implement Source FP8/MXFP4 Runtime Upload And Kernel Dispatch

**What:** Wire source `F8_E4M3_B128`, `MXFP4`, `BF16`, `F32`, and `I32` tensor
families into the V100 runtime without persistent F16 dequantized copies.

**Why:** Source inspect and manifest generation now work, but inference is
intentionally blocked until these execution paths exist.

**Severity:** Critical

**Suggested sprint:** Sprint 003

**Files:** `ds4.c`, CUDA/runtime files, kernel registry files

## 3. Add Source-Format Correctness Harness

**What:** Add a first-token or layer-slice harness for source-format decode once
the first execution path exists.

**Why:** The current source guard prevents accidental incorrect decode. The next
step needs a small, repeatable correctness gate before throughput testing.

**Severity:** Critical

**Suggested sprint:** Sprint 003

**Files:** `tests/`, `ds4.c`, future harness files

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Emit per-GPU shard files | Critical | Sprint 003 | `tools/`, future packer/runtime files |
| Implement source FP8/MXFP4 runtime upload and dispatch | Critical | Sprint 003 | `ds4.c`, CUDA/runtime files |
| Add source-format correctness harness | Critical | Sprint 003 | `tests/`, `ds4.c` |

