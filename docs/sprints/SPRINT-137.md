# Sprint 137 - 128-Slot 32K Throughput Admission

Date: 2026-05-21

## Objective

Map the next scheduler-side throughput ceiling after Sprint 136. Admit 128
slots only at a short 32K context so the experiment stays inside the 32 GB V100
memory budget, then compare against a same-context 64-slot control.

## Implementation

- Raised the scheduler slot ceiling from `64` to `128`.
- Raised the layer execution active-batch ceiling from `64` to `128`.
- Updated the appliance launcher admission policy:
  - `ctx <= 32768`: up to 128 slots.
  - `ctx <= 65536`: up to 64 slots.
  - `ctx <= 131072`: up to 32 slots.
  - `ctx <= 262144`: up to 16 slots.
  - `ctx <= 524288`: up to 14 slots.
  - `ctx > 524288`: up to 7 slots.
- Updated the soak harness to permit `--slots 128`.
- Raised the planner ceiling to 128 slots and added 32K/64K tiers to the
  context-admission table.

This keeps 128-slot serving explicit and short-context only. It does not
change the default production slot count.

## Validation

Local launcher checks:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-appliance-soak.sh
DS4_V100_CTX=32768 DS4_V100_SLOTS=128 DS4_V100_ACTIVE_MICROBATCH=128 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
DS4_V100_CTX=65536 DS4_V100_SLOTS=128 DS4_V100_ACTIVE_MICROBATCH=128 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
```

The 32K/128-slot config was accepted. The 64K/128-slot config was rejected with
the expected cap:

```text
DS4_V100_SLOTS=128 exceeds ctx=65536 admission cap 64
```

Cluster build forced stale header dependents to rebuild:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && rm -f tools/ds4-v100-replay.o tools/ds4-v100-replay \
   tests/cuda_v100_full_scheduler_smoke.o tests/cuda_v100_full_scheduler_smoke \
   ds4_v100_scheduler.o ds4_v100_layer_execute.o tools/ds4-v100-plan && \
   make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke \
   tools/ds4-v100-plan CUDA_ARCH=sm_70 -j80'
```

Full 43-layer smoke:

```text
./tests/cuda_v100_full_scheduler_smoke \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --ctx 32768 --slots 128
```

Result:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=128 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

The served `/status` and metrics confirmed the rebuilt process was actually
running the 128-slot binary:

```text
"mode":"base_slots_128"
"configured_slots":128
"active_microbatch":128
ds4_v100_configured_slots 128
ds4_v100_active_microbatch 128
ds4_v100_active_slots 128
ds4_v100_concurrent_request_capacity 128
```

Planner check:

```text
ds4-v100-plan --ctx 32768 --slots 128 --gpus 8 --reserve-gib 4 --mtp off
```

The configured 128-slot/32K plan fit with worst-GPU total `28.97 GiB / 32.00
GiB`, including the 4 GiB reserve. The memory-only tier table reports:

```text
32768 -> 128 slots, 28.97 GiB worst GPU
65536 -> 114 slots, 31.96 GiB worst GPU
131072 -> 57 slots, 31.91 GiB worst GPU
```

The launcher deliberately keeps simpler conservative caps below those memory
ceilings for 64K and 128K.

## Throughput

Both served runs used the current fused TurboMind appliance path with compact
expert scheduling enabled and route-row-reduce, indexed-A, and gated-SiLU
disabled.

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Sprint 137 128-slot throughput | 32,768 | 128 | `59.598172` | `55.873286` | 128/128 token match |
| Sprint 137 same-context control | 32,768 | 64 | `57.170428` | `53.597276` | 64/64 token match |

The 128-slot mode is about `4.2%` faster than the 64-slot same-context control.
That is a real top-line improvement, but the marginal gain is smaller than the
32-slot and 64-slot steps.

Sampled utilization showed the wider run did hit high instantaneous activity:
gpu0, gpu1, gpu2, gpu3, gpu4, and gpu7 each reached `95-97%` max sampled
utilization during the run, with gpu5/gpu6 lower because the current stage
layout and tail work are imbalanced.

## Decision

Ship the 128-slot 32K admission path as an explicit short-context throughput
mode, but treat it as the end of the obvious scheduler-width sweep.

Slot-width scaling is still positive but diminishing:

| Step | Same-context gain |
|---|---:|
| 16 -> 32 slots at 128K | about `15.4%` |
| 32 -> 64 slots at 64K | about `8.4%` |
| 64 -> 128 slots at 32K | about `4.2%` |

The next material sprint should shift back to the kernel path: a lower-level
software-pipelined packed MXFP4 routed expert probe that targets the compact
served route shapes rather than another host-side dispatch or admission change.
