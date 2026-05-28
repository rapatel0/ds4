# TEMP Status Report 423

Date: 2026-05-27

## Focus

Implement the next TP/EP-only rank-major conversion: post-attention FFN shared
and routed inputs should consume rank-major post-attention hidden state instead
of relying on device-0 normalized full-hidden copies.

No PP/layer-split work was done.

## Implementation

Added default-off controls:

```text
--routed-ffn-rank-major-input-gate
DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
```

When enabled, the post-attention FFN input path:

- NCCL-allgathers `d_post_attn_shard` into each rank's rank-major buffer.
- Builds the device-0 slot-major tensor only for existing router logits.
- Fills shared FFN gate/up inputs from rank-major hidden with fused stable RMS
  norm and F32-to-F16 conversion.
- Packs routed expert rows from rank-major hidden with the same stable RMS norm
  formulation as the reference `rms_norm_weight_rows_stable_kernel`.
- Keeps the old slot-major path as fallback.

The first route-packer version used a simple RMS sum and changed checksum. That
was fixed by switching the route packer to the max-scaled stable RMS formula.

## Build

V100 sm_70 build passed on `gpu-01`.

The host does not expose a normal `/usr/local/cuda`; I reconstructed a local
CUDA link view at:

```text
/localpool/ds4/cuda-12.2-link
```

using the existing MicroK8s CUDA/NCCL container snapshots. This is a cluster
tooling workaround, not part of the runtime path.

## Resident Layer 2 A/B

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint423-rankmajor-routed-ffn/resident-layer2-postffn-control-skipstats/
/localpool/ds4/workspace/logs/sprint423-rankmajor-routed-ffn/resident-layer2-rankmajor-routed-stable-skipstats/
```

| Metric | Control | Rank-major routed |
|---|---:|---:|
| rc | 0 | 0 |
| checksum | 4161861552 | 4161861552 |
| graph capture/replay | pass | pass |
| decode ms/step | 3.404288 | 3.283712 |
| slot-step tok/s | 2349.977403 | 2436.267315 |
| graph replay ms | 13.617152 | 13.134848 |

Resident result: correct and about `+3.7%` faster for this layer-2 graph.

## All-Layer Direct A/B

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint423-rankmajor-routed-ffn/full-postffn-control-scratch256/
/localpool/ds4/workspace/logs/sprint423-rankmajor-routed-ffn/full-rankmajor-routed-scratch256/
```

Shape:

```text
slots=8
ctx=262144
decode_steps=4
scratch=256 MiB
persistent graph replay=on
deferred NCCL=on
semantic skip stats=on
post-attention FFN input=on
```

| Metric | Control | Rank-major routed |
|---|---:|---:|
| rc | 0 | 0 |
| capture/replay | 43/43, 172/172 | 43/43, 172/172 |
| generated decode tok/s | 60.003725 | 63.465436 |
| continuation decode tok/s | 64.984487 | 70.809013 |
| wall generated tok/s | 13.290042 | 11.644355 |
| checksum | 2784282403 | 6289750090 |
| graph nodes | 236536 | 235332 |

All-layer result: structurally clean and decode-positive, but not promotable yet
because checksum diverges from layer 0 onward.

## Decision

Keep `--routed-ffn-rank-major-input-gate` default-off.

This is a useful direction, but it needs a parity/debug sprint before HTTP
promotion. The resident layer result proves the stable route packer can match a
single-layer graph. The all-layer divergence points to a remaining sequencing,
buffer-reuse, or all-layer shared-binding issue.

## Inconclusive Isolation

I attempted a one-step all-layer eager/non-graph isolation run:

```text
/localpool/ds4/workspace/logs/sprint423-rankmajor-routed-ffn/full-postffn-control-eager-step1/
```

It failed before producing a useful control comparison:

```text
tp_ep_model_router_route_plan_failed layer 0 rc 8
tp_hc_current_input_failed layer 0 rc 5
```

So the checksum divergence is still known only in the persistent graph
all-layer A/B, not isolated to eager execution.

## Next

1. Add a focused parity probe for post-attention FFN inputs:
   - compare slot-major `hc->d_ffn_normed`
   - compare rank-major shared gate/up half inputs
   - compare routed `r.d_a`
   - run at layer 0 and layer 2 under all-layer shared bindings
2. Resolve all-layer checksum divergence.
3. Re-run all-layer direct A/B.
4. Only then run HTTP selected-token/chat promotion.
