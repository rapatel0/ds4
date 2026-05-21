# Sprint 121 - 16-Slot 256K Throughput Mode

Date: 2026-05-21

## Objective

Test whether the current Volta HMMA and TurboMind grouped-expert paths benefit
from a full 16 active-slot token tile at the 256K context tier. This is the
first practical serving change after the failed single-token/row-pair fusion
probes: instead of making smaller per-slot kernels, expose more active rows to
the existing batched kernels.

## Implementation

1. Raise `DS4_V100_LAYER_MAX_BATCH` and `DS4_V100_SCHED_MAX_SLOTS` from 8 to
   16.
2. Lift appliance smoke/soak/benchmark wrapper slot validation to 16.
3. Update `ds4-v100-plan` admission reporting to evaluate 16-slot tiers.
4. Add `--ctx` to the stage/full scheduler smokes so 16-slot correctness can
   be validated at the intended 256K context instead of the default 1M context.
5. Add launcher admission guards:
   - `ctx <= 262144`: up to 16 slots.
   - `262144 < ctx <= 524288`: up to 14 slots.
   - `ctx > 524288`: up to 7 slots.

The launcher default remains unchanged; this sprint only makes 16-slot 256K
serving admissible and measured.

## Validation

Local/static:

- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh`
- `bash -n tools/ds4-v100-appliance-smoke.sh tools/ds4-v100-aggregate-throughput.sh tools/ds4-v100-sustained-decode-bench.sh tools/ds4-v100-production-deployment-gate.sh tools/ds4-v100-slot-context-envelope.sh`
- `make -j8 tools/ds4-v100-plan ds4_v100_replay.o`
- `./tools/ds4-v100-plan --ctx 262144 --slots 16 --gpus 8 --mtp off`

Cluster build:

- `CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-plan tests/cuda_v100_full_scheduler_smoke tests/cuda_v100_stage_scheduler_smoke tools/ds4-v100-replay`

Cluster correctness:

- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144`: passed.
- `cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43`: passed.
- Launcher config check accepts `ctx=262144 slots=16`.
- Launcher config check rejects `ctx=1048576 slots=16`.
- Launcher config check still accepts `ctx=1048576 slots=4`.

## Results

Same-binary served A/B on the 8x V100 node:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| 8-slot control | 262,144 | 8 | `34.445844` | `32.292979` | 8/8 token match |
| 16-slot candidate | 262,144 | 16 | `43.659461` | `40.930745` | 16/16 token match |

The 16-slot candidate improves generated throughput by about `26.7%` over the
same-binary 8-slot control. It also raises coarse sampled GPU utilization from
about `6.76%` to `9.07%` during the full soak window, while max observed memory
remains below 24 GiB in the sampled telemetry.

Artifacts:

- `logs/from-cluster/sprint121-16slot/control8/summary.json`
- `logs/from-cluster/sprint121-16slot/candidate16/summary.json`
- `logs/from-cluster/sprint121-16slot/control8/gpu_util.csv`
- `logs/from-cluster/sprint121-16slot/candidate16/gpu_util.csv`

## Decision

Keep 16-slot 256K as an admitted throughput mode. This is not a default for
long-context operation because 16 slots at 1M is not memory-safe; the launcher
now rejects that configuration before allocation.

This result supports the software-pipelining hypothesis: the existing HMMA and
TurboMind grouped paths improve when they see more active token rows. The next
larger optimization should either build a real SM70 pipelined F8 mainloop or
rework expert scheduling for even higher active route counts.
