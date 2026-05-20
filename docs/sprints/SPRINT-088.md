# Sprint 088: Scheduler-Bound TurboMind Appliance Runtime

## Goal

Wire the single-shard appliance format into the runtime path:

- Context opens `pack-index.tsv` for non-experts.
- Context opens `turbomind-pack-index.tsv` for routed experts.
- Layer state binds routed experts to TurboMind resident spans.
- Scheduler can map and upload `gpuN.weights` from an appliance directory.
- Layer execution dispatches the no-repack TurboMind routed FFN path.

## Done

- Added `turbomind_pack_index_path` to V100 context and scheduler options.
- Added `shard_dir` scheduler mode for appliance `gpuN.weights` loading.
- Added TurboMind descriptor lookup and layer binding APIs.
- Added shard-offset model-map mode for CPU-side control tensors in appliance
  shards.
- Added a context smoke covering TurboMind metadata, arena sizing, and binding
  lookup.

## Validation

Local CPU validation:

```text
make tests/v100_context_smoke ds4_v100_layer_execute.o ds4_v100_scheduler.o
./tests/v100_context_smoke
```

Result:

```text
v100_context_smoke: ok
```

V100 build validation:

```text
CUDA_ARCH=sm_70 make tests/v100_context_smoke ds4_v100_layer_execute.o ds4_v100_scheduler.o \
  tests/cuda_v100_turbomind_sidecar_smoke tools/ds4-v100-appliance-pack
CUDA_ARCH=sm_70 make tests/cuda_v100_stage_scheduler_smoke tests/cuda_v100_full_scheduler_smoke
```

Result: all listed targets built on `llamacpp-build-8gpu`, and
`./tests/v100_context_smoke` passed in the V100 pod.

## Remaining

Run the scheduler appliance path on V100 against a full or bounded appliance
directory, then measure correctness and decode throughput with TurboMind expert
dispatch active.
