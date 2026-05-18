# SPRINT-026 Followups

## Critical

1. **Add stage/layer HC divergence checkpoints**

   Compare CPU source-layout HC and V100 HC after layers 5, 11, 17, 23, 29,
   34, 39, and 42 for `short_reasoning_plain`. If the first divergent stage is
   broad, bisect within that stage.

   - **Why**: Output-head parity passes, but selected-token replay still fails.
   - **Suggested sprint**: Sprint 027.
   - **Files**: `ds4_v100_scheduler.c`, `ds4_v100_layer_execute.c`,
     `tests/cuda_v100_selected_token_smoke.c`, new checkpoint smoke.

2. **Add per-layer executor counters and failure-local reports**

   Record token id, position, layer id, router kind, selected expert ids,
   compressed-row counts, indexer top-k, and HC norm before/after each layer.

   - **Why**: The top-k mismatch shows a large upstream drift, but current logs
     only report final logits.
   - **Suggested sprint**: Sprint 027.
   - **Files**: `ds4_v100_layer_execute.h`, `ds4_v100_layer_execute.c`,
     `ds4_v100_scheduler.c`.

## High

3. **Make full-gate workspaces self-building by default**

   Either run `tools/ds4-v100-gate.sh --build` in fresh cluster workspaces or
   add a guard that fails early when required binaries are absent.

   - **Why**: Sprint 026 had one invalid full-gate attempt without `--build`.
   - **Suggested sprint**: Next maintenance sprint.
   - **Files**: `tools/ds4-v100-gate.sh`, cluster test docs.

4. **Parallelize resident stage uploads in tests and deployment**

   Open/upload independent stage schedulers concurrently or add a pack-load
   service path that loads all eight resident arenas in parallel.

   - **Why**: Prompt replay correctness smokes spend most wall time in
     serialized shard upload, not decode.
   - **Suggested sprint**: After selected-token localization, before serving.
   - **Files**: `ds4_v100_scheduler.c`, future appliance server/loader code.

## Deferred

5. **Serving, MTP, throughput**

   Keep deferred until the selected-token divergence is localized and fixed.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Stage/layer HC divergence checkpoints | Critical | Sprint 027 | `ds4_v100_scheduler.c`, `ds4_v100_layer_execute.c`, tests |
| Per-layer counters and reports | Critical | Sprint 027 | `ds4_v100_layer_execute.*`, `ds4_v100_scheduler.c` |
| Full-gate build guard | High | Maintenance | `tools/ds4-v100-gate.sh`, cluster docs |
| Parallel resident uploads | High | Before serving | scheduler/server loader |
| Serving, MTP, throughput | Deferred | After correctness | runtime/server |
