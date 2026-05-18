---
sprint: 008
title: Source Oracle Harness And V100 KV Admission Anchors
status: in_progress
date: 2026-05-18
---

# SPRINT-008 Report

## Current Verdict

`IN_PROGRESS`.

Phase 2, V100 F16 KV admission, is implemented and validated locally and on the
8x V100 pod. The sprint is not complete: oracle automation, guard hardening,
source dtype parity hardening, and the bounded CUDA source-format anchor remain.

## Shipped In This Slice

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

## Not Yet Done

- Phase 1: automated official-vector source oracle and guard regressions.
- Phase 3: MXFP4/source dtype parity hardening.
- Phase 4: bounded CUDA source-format anchor.
- Phase 5: final sprint report, follow-ups, and `VISION.md` update after the
  full Sprint 008 verdict.

## Sprint 009 Handoff Notes

The exact F16 KV accounting surface is now available for the future prefill and
compressed-KV execution sprint. Runtime allocation should consume these derived
stage budgets rather than reintroducing a coarse per-GPU KV estimate.
