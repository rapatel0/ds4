---
sprint: 026
title: Output-Head Divergence Localization
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-026-REPORT.md
followups: SPRINT-026-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-026: Output-Head Divergence Localization

## Overview

Sprint 026 narrows the Sprint 025 selected-token failure. The sprint adds a
top-k selected-token diagnostic and an isolated CPU-vs-V100 output-head parity
smoke that feeds a deterministic HC vector directly into the gpu7 output-head
path.

The result is clear: output-head collapse and BF16 vocab projection match the
CPU reference for the deterministic HC fixture. The official prompt still
selects newline-newline instead of `16`, so the remaining correctness gap is
upstream in the 43-layer scheduler body.

## Outcome Contract

- `SHIP`: output-head parity passes on V100 and selected-token top-k evidence
  is recorded.
- `EXTEND`: output-head parity fails or top-k diagnostics cannot run.
- `STOP`: V100 output-head path cannot execute or produces non-finite logits.

## Non-Goals

- No public server.
- No MTP.
- No throughput benchmark.
- No multi-slot scheduler.
- No selected-token correctness claim.

## Implementation

### Phase 1: Scheduler Diagnostic API

**Files:**
- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`

**Tasks:**
- [x] Add `ds4_v100_stage_scheduler_write_hc` for deterministic HC injection.
- [x] Add `ds4_v100_stage_scheduler_select_topk`.
- [x] Keep `ds4_v100_stage_scheduler_select_token` as a top-1 wrapper.

### Phase 2: Output-Head Parity Smoke

**Files:**
- `tests/cuda_v100_output_head_parity_smoke.c`
- `Makefile`
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Build a deterministic `[4 x 4096]` HC vector.
- [x] Compute CPU reference logits through `hc_head_fn`, `hc_head_scale`,
  `hc_head_base`, `output_norm.weight`, and BF16 `output.weight`.
- [x] Inject the same HC into the gpu7 stage scheduler.
- [x] Compare CPU and V100 top-5 token ids and logits.
- [x] Add the parity smoke to the appliance gate.

### Phase 3: Prompt Top-K Diagnostic

**Files:**
- `tests/cuda_v100_selected_token_smoke.c`

**Tasks:**
- [x] Add `--top-k N`.
- [x] Print token id, logit, and token bytes for each selected top-k entry.
- [x] Preserve explicit `--expected-token-hex` failure semantics.

## Outcome

`SHIP`.

The output-head adapter is no longer the leading explanation for the Sprint
025 mismatch. The next sprint should checkpoint HC after stage/layer
boundaries against a CPU source-layout reference to identify where the body
first diverges.

## Definition Of Done

- Output-head parity smoke builds locally and on `sm_70`.
- Output-head parity passes on the 8x V100 pod.
- Prompt replay top-k diagnostic records the official-vector mismatch.
- Gate wiring includes the new parity target.
- Follow-up work is explicitly scoped to upstream scheduler-body divergence.
