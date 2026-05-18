---
sprint: 025
title: Scheduler Output-Head Selected Token Surface
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-025-REPORT.md
followups: SPRINT-025-FOLLOWUPS.md
verdict: EXTEND
---

# SPRINT-025: Scheduler Output-Head Selected Token Surface

## Overview

Sprint 025 attaches the gpu7 output-head path to the resident scheduler. The
new path consumes final HC from the full stage chain, collapses it through
`hc_head_*`, applies `output_norm.weight`, runs BF16 `output.weight`, and
selects the top-1 token.

This sprint produced a runnable output-head selected-token surface, but it did
not reach official-vector correctness. The explicit short-vector oracle check
currently selects newline-newline instead of `16`.

## Outcome Contract

- `SHIP`: prompt replay through all 43 layers produces the official
  `short_reasoning_plain` selected token.
- `EXTEND`: output-head execution works and produces finite logits/top-1, but
  the selected token does not match the official/source oracle.
- `STOP`: output-head execution cannot run on gpu7 or produces non-finite
  logits.

## Non-Goals

- No public server.
- No MTP.
- No throughput benchmark.
- No multi-slot wavefront.
- No claim that the numerical path is source-oracle-correct.

## Implementation

### Phase 1: Scheduler Output-Head API

**Files:**
- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`

**Tasks:**
- [x] Bind `hc_head_fn`, `hc_head_base`, `hc_head_scale`,
  `output_norm.weight`, and `output.weight` in the scheduler.
- [x] Add `ds4_v100_stage_scheduler_select_token`.
- [x] Require the output-head-owning stage.
- [x] Run HC collapse, output norm, BF16 output-head matmul, host argmax, and
  non-finite logit checks.

### Phase 2: Prompt Replay Smoke

**Files:**
- `tests/cuda_v100_selected_token_smoke.c`
- `Makefile`
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Tokenize the short official prompt through the existing DS4 tokenizer.
- [x] Replay all prompt tokens through the resident 8-stage scheduler.
- [x] Select top-1 from gpu7 output-head logits.
- [x] Make oracle comparison explicit via `--expected-token-hex`.
- [x] Add the output-head smoke to the appliance gate without removing the
  `real_model_selected_token` readiness blocker.

## Outcome

`EXTEND`.

The output-head path runs successfully on the V100 cluster and produces a
finite selected token. The explicit oracle check currently fails, so readiness
must still list `real_model_selected_token`.

## Definition Of Done

- Output-head selected-token smoke builds locally and on `sm_70`.
- V100 output-head smoke runs through prompt replay and emits a selected token.
- Explicit oracle comparison records the mismatch instead of being treated as a
  pass.
- Gate readiness remains honest about `real_model_selected_token`.
