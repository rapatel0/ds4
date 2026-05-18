# SPRINT-009 Follow-Ups

These are not blockers for the Sprint 009 `SHIP` verdict, but they should
shape Sprint 010+.

## Runtime Integration

- **What:** Replace the diagnostic host-F32 input row in
  `ds4_gpu_v100_prefill_kv_update_f16_tensor` with actual V100 projection and
  compressor outputs.
- **Why:** Sprint 009 proved the F16 KV write/update surfaces, not production
  dense/compressor math.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 010.
- **Files:** `ds4_cuda.cu`, `ds4_gpu.h`, future layer-scheduler files.

## Oracle Comparison

- **What:** Add a bounded V100-vs-source-oracle comparison for one short prompt
  slice after the real projection/compressor path is wired.
- **Why:** The current CUDA smoke uses synthetic deterministic state references;
  source-layout correctness still needs a model-derived comparison before
  serving unlock.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 010.
- **Files:** `tools/ds4-source-oracle-vector.c`, V100 prefill/decode tests.

## State Views

- **What:** Introduce explicit subviews for `attn_comp_kv_state`,
  `attn_comp_score_state`, `indexer_comp_kv_state`, and
  `indexer_comp_score_state` inside the stage-local KV arena.
- **Why:** Sprint 009 reports and allocates combined stage-local state bytes;
  production kernels need named views to avoid offset mistakes.
- **Severity:** Important.
- **Suggested sprint:** Sprint 010.
- **Files:** `ds4_v100_context.[ch]`, `ds4_v100_context_cuda.cu`.

## Deployment Sequencing

- **What:** Re-scope the next sprint from broad server deployment to a V100
  single-slot decode/prefill integration gate if full logits are still absent.
- **Why:** Sprint 009 shipped bounded KV execution, but not full layer execution
  or selected-token correctness on V100.
- **Severity:** Important.
- **Suggested sprint:** Sprint 010 planning.
- **Files:** `docs/sprints/VISION.md`, next sprint document.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Runtime integration | Critical | Sprint 010 | `ds4_cuda.cu`, `ds4_gpu.h`, scheduler files |
| Oracle comparison | Critical | Sprint 010 | `tools/ds4-source-oracle-vector.c`, V100 tests |
| State views | Important | Sprint 010 | `ds4_v100_context.[ch]`, `ds4_v100_context_cuda.cu` |
| Deployment sequencing | Important | Sprint 010 planning | `docs/sprints/VISION.md`, next sprint |
