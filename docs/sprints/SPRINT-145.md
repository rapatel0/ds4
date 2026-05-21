# Sprint 145 - 256-Slot Short-Context Admission

Date: 2026-05-21

## Objective

Test whether practical short-context serving can keep improving by widening the
configured slot ceiling beyond the Sprint 137 128-slot/32K mode. This sprint
targets a memory-safe 256-slot/16K tier and records prompt/prefill and
continuation/decode throughput separately.

## Implementation

- Raised the scheduler and layer batch ceilings from 128 to 256 slots.
- Raised the appliance soak harness `--slots` validation to 256.
- Extended the V100 planner to model 16K context and 256-slot admission.
- Added launcher admission for 256 slots only at `ctx <= 16384`; 32K remains
  capped at 128 slots even though the planner has some theoretical headroom.
- Kept the 200 ms high-slot rendezvous policy for throughput serving.

## V100 Validation

Build:

```text
make -j80 CUDA_ARCH=sm_70 \
  tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke tools/ds4-v100-plan
```

Planner result for `ctx=16384`, `slots=256`, 8 GPUs, 4 GiB reserve, MTP off:

```text
Worst configured GPU total including reserve: 29.07 GiB / 32.00 GiB
ctx=16384 max admitted slots: 256
ctx=32768 planner fit: 226 slots, but launcher cap remains 128
```

Full 43-layer scheduler smoke passed:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=256 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

Served 16K throughput curve with split metrics:

| Slots | Generated tok/s | Prompt tok/s | Continuation tok/s | Avg latency ms | Correctness |
|---:|---:|---:|---:|---:|---|
| 128 | `59.860493` | `67.343055` | `56.119213` | `34016.025` | 128/128 |
| 192 | `60.700926` | `68.288542` | `56.907118` | `50279.848` | 192/192 |
| 256 | `61.065087` | `68.698223` | `57.248519` | `66646.170` | 256/256 |

GPU utilization telemetry during the 256-slot run reached `96-98%` max on the
heaviest stages and stayed within the measured VRAM envelope. Average
run-level utilization is lower because the CSV includes model load, warmup, and
shutdown, but the active request batch does exercise the GPUs.

Artifacts:

- `logs/from-cluster/sprint145-plan-16k-256.txt`
- `logs/from-cluster/sprint145-served-128slot-16k/`
- `logs/from-cluster/sprint145-served-192slot-16k/`
- `logs/from-cluster/sprint145-served-256slot-16k/`

## Decision

Ship the 256-slot short-context admission as a guarded ceiling for `ctx <= 16K`.
Do not treat it as the main throughput strategy. The best 256-slot run improves
aggregate continuation/decode throughput by only about 2.0% over the 128-slot
16K control while roughly doubling per-request latency.

The data says simple slot widening is now mostly exhausted. The next material
performance sprint should attack a larger routed-FFN execution boundary:
software-pipelined packed MXFP4 dequant/HMMA/gated activation/down/reduce, or a
persistent expert scheduler that keeps the V100 tensor cores fed without losing
the current stage overlap.
