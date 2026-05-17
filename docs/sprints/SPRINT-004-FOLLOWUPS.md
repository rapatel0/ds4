# SPRINT-004 Follow-Ups

## Model-Less Default Test Target

**What:** Add a default test target that runs parser/unit/smoke tests without
requiring `ds4flash.gguf`, or split the existing model-dependent tests behind
an explicit live-model target.

**Why:** During Sprint 004 validation, `make test` on the laptop failed before
the new code path because the default model file `ds4flash.gguf` was not
present. The focused Sprint 004 tests passed, but the default test target is
not usable on a clean checkout without a model.

**Severity:** Important.

**Suggested sprint:** Next.

**Files:** `Makefile`, `tests/ds4_test.c`.

## Direct CUDA Arena Unit Target

**What:** Add a CUDA-linked arena unit target that directly exercises
`ds4_gpu_arena_open`, upload, readback, memory-kind reporting, and invalid
range failure on a real GPU.

**Why:** Sprint 004 exercised the CUDA arena path through the full residency
smoke and used `tests/gpu_arena_smoke` for CPU/stub behavior. A smaller direct
CUDA test would catch arena regressions faster than a full 145 GiB residency
run.

**Severity:** Nice-to-have.

**Suggested sprint:** Sprint 005.

**Files:** `Makefile`, `tests/gpu_arena_smoke.c`, `ds4_cuda.cu`.

## Upload Timing Metrics

**What:** Add elapsed-time and throughput rows to
`tools/ds4-v100-residency-smoke` for each provider and each GPU.

**Why:** Sprint 004 proved correctness and residency. It did not record
provider throughput, so future upload optimization work lacks a baseline.

**Severity:** Nice-to-have.

**Suggested sprint:** Future.

**Files:** `tools/ds4-v100-residency-smoke.c`, `docs/sprints/drafts/*`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Model-less default test target | Important | Next | `Makefile`, `tests/ds4_test.c` |
| Direct CUDA arena unit target | Nice-to-have | Sprint 005 | `Makefile`, `tests/gpu_arena_smoke.c`, `ds4_cuda.cu` |
| Upload timing metrics | Nice-to-have | Future | `tools/ds4-v100-residency-smoke.c` |
