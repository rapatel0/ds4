# Sprint 089: Appliance-Backed Scheduler Smoke

## Goal

Execute a V100 scheduler smoke from an appliance directory instead of the
source GGUF map.

## Implementation Plan

- Let `ds4-v100-appliance-pack --layer N` produce a hybrid bounded appliance:
  the selected layer's routed experts are TurboMind-packed, while unselected
  routed experts are copied in source format unless `--skip-non-experts` is set.
- Add `--shard-dir` and `--tm-index` to scheduler smoke binaries.
- Generate a bounded stage-0 appliance with layer-0 TurboMind experts and the
  rest of the tensors source-packed.
- Run `cuda_v100_stage_scheduler_smoke` on V100 with `--shard-dir` and
  `--tm-index`.

## Definition Of Done

- [x] Hybrid appliance pack generation succeeds on V100.
- [x] Scheduler smoke opens the appliance directory without the source GGUF
  model map.
- [x] Stage 0 executes assigned layers and returns finite nonzero HC output.
- [x] Validation log is committed.

## Result

Sprint 089 proves the appliance scheduler path on V100 for a bounded stage-0
pack:

```text
ds4-v100-appliance-pack: gpu0.weights bytes=22524134668
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 tm_layers=1 ... ok
cuda_v100_full_scheduler_smoke: stages=1 ... layers=6 tm_layers=1 ... ok
```

The `tm_layers=1` report is the positive assertion that layer 0 dispatched
through the no-repack TurboMind routed expert path inside scheduler execution.

## Non-Goals

- Full all-layer TurboMind appliance generation.
- Aggregate throughput benchmark.
- MTP throughput optimization.
