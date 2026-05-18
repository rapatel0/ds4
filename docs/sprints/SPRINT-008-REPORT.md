---
sprint: 008
title: Source Oracle Harness And V100 KV Admission Anchors
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-008 Report: Source Oracle Harness And V100 KV Admission Anchors

## Verdict

`SHIP`

Sprint 008 turned the Sprint 007 one-off source-layout oracle into repeatable
validation, added exact DS4-aware F16 KV admission for the layer-owned V100
topology, hardened source-format helpers, and landed a bounded CUDA
F8_E4M3_B128 source-format anchor on `sm_70`.

Normal source-layout generation remains fail-closed.

## What Shipped

- Added DS4 layer-class metadata for SWA-only, ratio-4/indexer, and ratio-128
  layers.
- Added F16 KV budget fields for raw SWA, compressed attention KV, ratio-4
  indexer KV, compression-state reserve, and total planned KV bytes.
- Derived per-stage KV budgets from `kv_ctx_tokens`, `kv_active_slots`, and the
  existing layer-owned V100 topology.
- Added reserve admission checks that fail closed when derived KV would overfill
  a V100 stage.
- Extended model-less and CUDA V100 context smoke entrypoints with `--kv-ctx`
  and `--kv-slots`.
- Printed stable report fields for context, active slots, layer-class counts,
  per-stage KV bytes, and per-layer KV class/bytes.
- Added `tools/ds4-source-oracle-vector` for automated official-vector source
  oracle validation and guard checks.
- Documented the source oracle command and `--dry-parse`/`--guard-checks`
  options in `tests/test-vectors/README.md`.
- Added packed-row span validators for F8_E4M3_B128 and MXFP4 source helpers.
- Added direct MXFP4 GGML `block_mxfp4` low-half/high-half ordering regression
  coverage and undersized packed-row span tests.
- Added a bounded `ds4_gpu_arena_f8_e4m3_b128_row_decode_f32` diagnostic API in
  the CUDA arena path plus a host stub implementation.
- Added `tests/cuda_source_dtypes_smoke.c`, which uploads strided synthetic
  F8_E4M3_B128 rows, decodes selected rows on CUDA, compares them to
  `ds4_source_formats`, and rejects invalid row ids, output spans, row strides,
  column counts, and truncated row views.

## Evidence

Local validation:

- `docs/sprints/drafts/SPRINT-008-V100-CONTEXT-SMOKE.log`
  - `v100_context_smoke: ok`
- `docs/sprints/drafts/SPRINT-008-KV-ADMISSION.log`
  - Covers 128K, 256K, 512K, and 1M single-slot KV tiers.
- `docs/sprints/drafts/SPRINT-008-CPU-BUILD.log`
  - `make -B cpu`
- `docs/sprints/drafts/SPRINT-008-CLI-BUILD.log`
  - `make -B ds4`
- `docs/sprints/drafts/SPRINT-008-DIFF-CHECK.log`
  - Empty output from `git diff --check`.
- `docs/sprints/drafts/SPRINT-008-LOCAL-PHASE1-3.log`
  - Builds `tools/ds4-source-oracle-vector` and `tests/source_dtypes_smoke`.
  - `tools/ds4-source-oracle-vector --dry-parse --only short_reasoning_plain`
    parses the selected token bytes.
  - `source_dtypes_smoke: ok`.
- `docs/sprints/drafts/SPRINT-008-PHASE1-3-DIFF-CHECK.log`
  - Empty output from `git diff --check`.
- `docs/sprints/drafts/SPRINT-008-SOURCE-DTYPES.log`
  - `source_dtypes_smoke: ok`.
- `docs/sprints/drafts/SPRINT-008-CUDA-SOURCE-LOCAL-COMPILE.log`
  - Local C compile check for `tests/cuda_source_dtypes_smoke.o`.
- `docs/sprints/drafts/SPRINT-008-CUDA-SOURCE-DIFF-CHECK.log`
  - Empty output from `git diff --check`.
- `docs/sprints/drafts/SPRINT-008-FINAL-LOCAL.log`
  - Final local build of the oracle tool, source dtype smoke, CUDA source test
    object, CPU binaries, and default `ds4`.
  - Final dry parse and `source_dtypes_smoke: ok`.
- `docs/sprints/drafts/SPRINT-008-FINAL-DIFF-CHECK.log`
  - Empty output from `git diff --check`.
- `docs/sprints/drafts/SPRINT-008-GUARDS-ONLY-LOCAL.log`
  - Builds `tools/ds4-source-oracle-vector` and dry-parses the short vector
    after adding `--guards-only`.
- `docs/sprints/drafts/SPRINT-008-GUARDS-ONLY-DIFF-CHECK.log`
  - Empty output from `git diff --check`.

Cluster validation:

- `docs/sprints/drafts/SPRINT-008-cluster-logs/v100-context-smoke.log`
  - Pod sees 8 `Tesla V100-SXM2-32GB` devices.
  - `v100_context_smoke: ok`.
  - 1M, single-slot derived KV by stage is approximately 0.63-0.96 GiB.
- `docs/sprints/drafts/SPRINT-008-cluster-logs/cuda-v100-context-smoke.log`
  - Built `tests/cuda_v100_context_smoke` with `CUDA_ARCH=sm_70`.
  - Verified `devices=8 stages=8 production=1`.
  - Device memory is `34072559616` bytes per V100.
  - Peer access matrix is fully enabled.
  - `cuda_v100_context_smoke: ok`.
- `docs/sprints/drafts/SPRINT-008-cluster-logs/cuda-v100-kv-overbudget.log`
  - `--kv-ctx 1048576 --kv-slots 64` fails closed:
    `stage 0 falls below reserve`.
- `docs/sprints/drafts/SPRINT-008-cluster-logs/source-oracle-and-dtypes.log`
  - Builds `tools/ds4-source-oracle-vector` and `tests/source_dtypes_smoke`.
  - `source_dtypes_smoke: ok`.
  - Guard checks pass for normal source-layout rejection, non-CPU oracle
    rejection, MTP oracle rejection, and missing diagnostic session unlock.
  - `vector short_reasoning_plain OK selected=3136 token=926`.
- `docs/sprints/drafts/SPRINT-008-ORACLE.log`
  - Condensed source-oracle selected-token evidence.
- `docs/sprints/drafts/SPRINT-008-GUARD.log`
  - Condensed source-layout guard evidence.
- `docs/sprints/drafts/SPRINT-008-CUDA-SOURCE.log`
  - Builds `tests/cuda_source_dtypes_smoke` with `CUDA_ARCH=sm_70`.
  - `cuda_source_dtypes_smoke: ok`.
- `docs/sprints/drafts/SPRINT-008-GUARDS-ONLY-CLUSTER.log`
  - Builds `tools/ds4-source-oracle-vector` and verifies
    `--guards-only` against `/models/DSv4-Flash-256e-fixed.gguf`.

## Deviations

- The CUDA source-format anchor uses F8_E4M3_B128 row decode rather than a
  row-dot. This is the narrower anchor and directly supports the next dense
  source-format path.
- The oracle harness lives in a dedicated tool instead of broadening
  `tests/ds4_test.c`. This keeps the real-model requirement and diagnostic
  session unlock out of model-less default tests.
- The CUDA anchor is synthetic and diagnostic-only. It is intentionally not
  wired into decode, prefill, or production serving.

## Remaining Scope

The project is still not deployed or performance optimized. Sprint 009 should
consume these contracts to start real V100 prompt prefill and compressed-KV
state population:

- source oracle runner and guard checks as correctness gates;
- exact F16 KV admission by context/slot/stage;
- F8_E4M3_B128 device row-decode parity as the first source-format CUDA anchor;
- MXFP4 low-half/high-half CPU parity as the routed expert layout reference.

## Sprint 009 Handoff Notes

The exact F16 KV accounting surface is now available for the future prefill and
compressed-KV execution sprint. Runtime allocation should consume these derived
stage budgets rather than reintroducing a coarse per-GPU KV estimate.
