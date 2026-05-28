# TEMP Status Report 424

Date: 2026-05-27

## Focus

Continue the TP/EP-only rank-major routed FFN input work from Sprint 423 and
resolve the all-layer checksum divergence before any HTTP promotion.

No PP/layer-split work was done.

## Code Change

Split the rank-major scratch used by HC-current input from the rank-major
scratch used by post-attention FFN input:

```text
RankState::d_current_full_rank_major
RankState::d_post_attn_full_rank_major
```

The new post-attention buffer is used only for the
`--routed-ffn-rank-major-input-gate` path. It costs about
`slots * hidden * sizeof(float)` per GPU, roughly `2 MiB/GPU` at `32` slots,
so it is not material to the VRAM plan.

## Build

V100 sm_70 build passed on `gpu-01` using:

```text
/localpool/ds4/cuda-12.2-link
```

## Resident Layer Parity

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint424-rankmajor-routed-parity/
```

| Case | Control checksum | Rank-major checksum | Control ms/step | Rank-major ms/step | Result |
|---|---:|---:|---:|---:|---|
| Resident layer 0 | 4710513124 | 4710513124 | 2.687232 | 2.531072 | Match |
| Resident layer 1 | 2210688361 | 2210688361 | 2.970880 | 2.496768 | Match |
| Resident layer 2 | 4161861552 | 4161861552 | 3.344384 | 3.225600 | Match |

This confirms the local rank-major FFN input math is correct for the first
three resident-layer graph shapes tested.

## All-Layer Dedicated-Buffer A/B

Shape:

```text
slots=8
ctx=262144
decode_steps=4
persistent graph replay=on
deferred NCCL=on
semantic skip stats=on
post-attention FFN input=on
```

| Metric | Control | Rank-major |
|---|---:|---:|
| rc | 0 | 0 |
| generated decode tok/s | 59.211511 | 63.430526 |
| continuation decode tok/s | 65.529013 | 70.936099 |
| checksum | 353694659 | 46803184 |
| first differing item | - | step 0, layer 1 |

First per-layer diff:

```text
control   step=0 layer=1 ratio=0 checksum=4621600222
rankmajor step=0 layer=1 ratio=0 checksum=276595468
```

The dedicated buffer did not restore all-layer parity.

## Serial EP/Dense Isolation

I also ran a one-step all-layer persistent graph pair with
`--serial-ep-dense`.

| Metric | Control | Rank-major |
|---|---:|---:|
| rc | 0 | 0 |
| generated decode tok/s | 43.977911 | 45.645493 |
| checksum | 2302003765 | 1799307679 |
| first differing item | - | step 0, layer 0 |

This did not prove an overlap-only bug. In fact, the serial path diverged
earlier than the overlapped path.

## Current Assessment

Do not promote `--routed-ffn-rank-major-input-gate`.

The rank-major route/shared input kernels are locally correct in resident
layer tests, and they remain decode-positive in all-layer timing. The remaining
blocker is all-layer graph/state semantics: the graph-captured full run does
not preserve the same carried state/checksum as the slot-major control.

Most likely next probes:

1. Split the gate into shared-only and routed-only rank-major inputs.
2. Add all-layer parity counters for:
   - shared gate/up input half tensors
   - routed `r.d_a`
   - `d_next_hidden`
   - `d_final_hc_shard`
3. Run those probes at layers 0 and 1 under the full all-layer persistent graph
   harness, not just resident-layer mode.

## Cluster State

All probe processes were stopped or completed. Final check showed all GPUs idle
with `0 MiB` allocated.
