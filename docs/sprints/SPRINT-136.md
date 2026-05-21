# Sprint 136 - 64-Slot 64K Throughput Admission

Date: 2026-05-21

## Objective

Test whether active-slot width continues to improve aggregate throughput after
Sprint 135 proved 32-slot 128K serving. Keep the admission policy conservative:
64 slots are allowed only at 64K context, while 128K stays capped at 32 slots
and 256K stays capped at 16 slots.

## Implementation

- Raised the scheduler slot ceiling from `32` to `64`.
- Raised the layer execution active-batch ceiling from `32` to `64`.
- Updated the appliance launcher admission policy:
  - `ctx <= 65536`: up to 64 slots.
  - `ctx <= 131072`: up to 32 slots.
  - `ctx <= 262144`: up to 16 slots.
  - `ctx <= 524288`: up to 14 slots.
  - `ctx > 524288`: up to 7 slots.
- Updated the soak harness to permit `--slots 64`.

This keeps 64-slot serving explicit and short-context only. It does not change
the default production slot count.

## Validation

Local launcher checks:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-appliance-soak.sh
DS4_V100_CTX=65536 DS4_V100_SLOTS=64 DS4_V100_ACTIVE_MICROBATCH=64 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
DS4_V100_CTX=131072 DS4_V100_SLOTS=64 DS4_V100_ACTIVE_MICROBATCH=64 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
```

The 64K/64-slot config was accepted. The 128K/64-slot config was rejected with
the expected cap:

```text
DS4_V100_SLOTS=64 exceeds ctx=131072 admission cap 32
```

Cluster build forced stale header dependents to rebuild:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && rm -f tools/ds4-v100-replay.o tools/ds4-v100-replay \
   tests/cuda_v100_full_scheduler_smoke.o tests/cuda_v100_full_scheduler_smoke \
   ds4_v100_scheduler.o ds4_v100_layer_execute.o && \
   make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke \
   CUDA_ARCH=sm_70 -j80'
```

Full 43-layer smoke:

```text
./tests/cuda_v100_full_scheduler_smoke \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --ctx 65536 --slots 64
```

Result:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=64 \
layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 \
uploaded_bytes=156142896212 expert_last=26 ok
```

## Throughput

Both served runs used the current fused TurboMind appliance path with compact
expert scheduling enabled and route-row-reduce, indexed-A, and gated-SiLU
disabled.

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Sprint 136 64-slot throughput | 65,536 | 64 | `57.322945` | `53.740261` | 64/64 token match |
| Sprint 136 same-context control | 65,536 | 32 | `52.884400` | `49.579125` | 32/32 token match |

The 64-slot mode is about `8.4%` faster than the 32-slot same-context control.
The 32-slot/64K control is effectively equal to Sprint 135's 32-slot/128K
result, so the improvement is from wider active-slot scheduling rather than
the shorter configured context.

Sampled request-window GPU utilization was materially higher than earlier
low-util runs: gpu0 reached about `93.5%` tail-window average, middle stages
were roughly `76-83%`, and gpu7 remained lower because the output/MTP tail is
still imbalanced.

## Decision

Ship the 64-slot 64K admission path as an explicit short-context throughput
mode.

This is the best measured served result so far, but still far below the
vision target. Wider slot admission is now proven useful but shows diminishing
returns: Sprint 135's 32-slot step added about `15.4%`, while Sprint 136's
64-slot step added about `8.4%`. The next material work should either test an
even shorter synthetic 96/128-slot ceiling for occupancy mapping or move into
the lower-level software-pipelined packed MXFP4 expert kernel path.
