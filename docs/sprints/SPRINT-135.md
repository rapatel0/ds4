# Sprint 135 - 32-Slot 128K Throughput Admission

Date: 2026-05-21

## Objective

Increase practical aggregate serving throughput by admitting wider active
microbatches where the V100 memory budget allows it. Keep 256K and longer
contexts capped conservatively, but allow 32-slot throughput serving at 128K.

## Implementation

- Raised the scheduler slot ceiling from `16` to `32`.
- Raised the layer execution active-batch ceiling from `16` to `32`.
- Updated the appliance launcher admission policy:
  - `ctx <= 131072`: up to 32 slots.
  - `ctx <= 262144`: up to 16 slots.
  - `ctx <= 524288`: up to 14 slots.
  - `ctx > 524288`: up to 7 slots.
- Updated the soak harness to permit `--slots 32`.

The implementation keeps the wider mode opt-in through explicit slot/context
configuration; it does not change the default production slot count.

## Validation

Local launcher checks:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-appliance-soak.sh
DS4_V100_CTX=131072 DS4_V100_SLOTS=32 DS4_V100_ACTIVE_MICROBATCH=32 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
DS4_V100_CTX=262144 DS4_V100_SLOTS=32 DS4_V100_ACTIVE_MICROBATCH=32 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
```

The 128K/32-slot config was accepted. The 256K/32-slot config was rejected
with the expected cap:

```text
DS4_V100_SLOTS=32 exceeds ctx=262144 admission cap 16
```

Cluster build:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && make tools/ds4-v100-replay \
   tests/cuda_v100_full_scheduler_smoke CUDA_ARCH=sm_70 -j80'
```

Full 43-layer smoke:

```text
./tests/cuda_v100_full_scheduler_smoke \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --ctx 131072 --slots 32
```

Result:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=32 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

## Throughput

Both runs used the current fused TurboMind appliance path with compact expert
scheduling enabled and route-row-reduce, indexed-A, and gated-SiLU disabled.

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Sprint 135 32-slot throughput | 131,072 | 32 | `52.840889` | `49.538334` | 32/32 token match |
| Sprint 135 same-context control | 131,072 | 16 | `45.780913` | `42.919606` | 16/16 token match |

The 32-slot mode is about `15.4%` faster than the 16-slot same-context control.
During the request window, sampled GPU utilization was much higher than the
earlier low-util observations, with the hottest GPUs around the high 60s to low
70s percent and the tail stage lower because of the current layer/MTP/output
imbalance.

## Decision

Ship the 32-slot 128K admission path as an explicit throughput mode.

This is not enough for the vision target, but it is a real practical serving
gain and confirms that wider active-slot scheduling is currently a stronger
lever than wrapper-level TurboMind dispatch changes. The next scaling step is
to test whether a 64-slot short-context mode fits and whether the resulting
expert microbatches improve GPU occupancy further.
