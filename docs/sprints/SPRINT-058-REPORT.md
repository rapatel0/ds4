# Sprint 058 Report: Replay Router Readback Suppression

## Result

`SHIP`.

## Changes Implemented

1. Added router-readback suppression as an explicit runtime option.
   - `ds4_v100_replay_options` now defaults replay generation to suppress
     router selected-expert and route-weight CPU readbacks.
   - `ds4_v100_stage_scheduler_options` propagates that setting to every stage.
   - `ds4_v100_layer_execute_config` controls the layer-level behavior.
2. Kept direct diagnostics intact by default.
   - Scheduler and layer tests that instantiate options directly still get the
     previous readback, selected-expert validation, and report population.
   - Only the replay appliance default skips readback.
3. Updated both single-token and batched FFN paths.
   - `execute_ffn_delta` skips selected/weight tensor reads when suppression is
     enabled.
   - `execute_ffn_delta_batch` skips the batched selected/weight readback only
     when every active slot has suppression enabled.
   - Reports still expose the route count; selected ids and weights are zeroed
     when readback is suppressed.

## Validation

Local:

```bash
cc -fsyntax-only -I. ds4_v100_layer_execute.c
cc -fsyntax-only -I. ds4_v100_scheduler.c
cc -fsyntax-only -I. ds4_v100_replay.c
cc -fsyntax-only -I. tools/ds4-v100-replay.c
make ds4_v100_layer_execute.o ds4_v100_scheduler.o ds4_v100_replay.o tools/ds4-v100-replay.o
git diff --check
```

Cluster build:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_ARCH=sm_70 make tools/ds4-v100-replay \
    tests/cuda_v100_integrated_layer_smoke \
    tests/cuda_v100_stage_scheduler_smoke
'
```

Focused V100 smokes:

```text
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 ... ok
cuda_v100_integrated_layer_smoke: layer=2 token=16 pos=16 ... ok
```

Note: the first integrated-layer smoke was launched concurrently with the stage
scheduler smoke on the same visible GPU and failed arena allocation with OOM.
The same integrated-layer smoke passed when rerun alone.

Real replay correctness: first token id `926`, text `16`, hex `3136`.

## Sustained Decode Comparison

Default path:

| Build | slots | generated tok/s | continuation tok/s | avg GPU util | max GPU util |
|---|---:|---:|---:|---:|---:|
| Sprint 057 default | 1 | 3.560863 | 3.338309 | 10.655% | 20.000% |
| Sprint 058 default | 1 | 3.583987 | 3.359988 | 11.141% | 20.000% |
| Sprint 057 default | 2 | 3.662490 | 3.433585 | 10.756% | 20.000% |
| Sprint 058 default | 2 | 3.704572 | 3.473036 | 11.162% | 20.000% |

Sprint 058 improved default generated tok/s by about `0.65%` at one slot and
about `1.15%` at two slots. This confirms the readback synchronization was real
overhead, but not the dominant bottleneck.

Artifacts:

- `logs/from-cluster/sprint058-router-readback-suppression/replay.json`
- `logs/from-cluster/sprint058-router-readback-suppression/sustained_decode.tsv`
- `logs/from-cluster/sprint058-router-readback-suppression/sustained_decode.json`
- per-case `result.json`, `server_status_before.json`,
  `server_status_after.json`, `server.log`, and `gpu_util.csv`

## Assessment

Suppressing replay-only router readback is a correct hot-path cleanup and gives
a small measurable speedup while preserving diagnostic defaults. It does not
materially change the utilization picture: the appliance is still near `11%`
average GPU utilization on the measured sustained decode path.

The next sprint should not spend more time on report/readback cleanup. The
remaining practical-use bottleneck is still the low-occupancy MoE/layer shape:
per-layer allocation/copy overhead in the opt-in batch path, no copy-free
active-slot layout, and no persistent or tensor-core-friendly grouped expert
kernel.
