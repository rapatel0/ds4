# Sprint 423: Rank-Major Routed FFN Input Packing

## Objective

Continue the TP/EP-only rank-major conversion by feeding shared and routed MoE
expert inputs from rank-major post-attention hidden state instead of from
slot-major full-hidden tensors.

No PP/layer-split work is in scope.

## Rationale

The current post-attention FFN input path still packs shared and expert input
rows from `r.d_current_full`, a slot-major normalized full-hidden tensor copied
out from device 0. That preserves correctness, but it keeps a downstream
consumer coupled to full-hidden slot-major staging.

This sprint adds a default-off route packer that reads:

```text
[rank][slot][hidden / 8]
```

and writes routed expert rows directly:

```text
[route][hidden]
```

## Implementation

Add:

```text
--routed-ffn-rank-major-input-gate
DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
```

When active:

- require/enable NCCL allgather
- allgather post-attention shards into rank-major buffers
- fill shared FFN gate/up inputs from rank-major post-attention hidden
- pack `r.d_a` from rank-major post-attention hidden
- apply stable RMS norm in the rank-major shared/route packers
- preserve the existing scaled route path for reference guard mode
- keep the existing slot-major packer as fallback

## Definition of Done

- V100 sm_70 build passes.
- Resident layer 2 passes with unchanged checksum and graph capture/replay.
- Full all-layer direct decode at `8` slots / `256K` / `4` decode steps passes.
- Status report records whether this reduced graph nodes or improved tok/s.
- Vision decision log is updated.

## Outcome

Status: implemented and partially validated; not promoted.

V100 sm_70 build passed. The build used a temporary CUDA link view at
`/localpool/ds4/cuda-12.2-link` because the host exposes CUDA only through
MicroK8s container snapshots.

Resident layer 2 A/B:

| Metric | Control | Rank-major routed |
|---|---:|---:|
| rc | 0 | 0 |
| checksum | 4161861552 | 4161861552 |
| decode ms/step | 3.404288 | 3.283712 |
| slot-step tok/s | 2349.977403 | 2436.267315 |

All-layer direct A/B at `8` slots / `256K` / `4` decode steps:

| Metric | Control | Rank-major routed |
|---|---:|---:|
| rc | 0 | 0 |
| capture/replay | 43/43, 172/172 | 43/43, 172/172 |
| generated decode tok/s | 60.003725 | 63.465436 |
| continuation decode tok/s | 64.984487 | 70.809013 |
| checksum | 2784282403 | 6289750090 |
| graph nodes | 236536 | 235332 |

Decision:

- Keep `--routed-ffn-rank-major-input-gate` default-off.
- The resident layer validates the stable rank-major packer.
- The all-layer checksum divergence starts at layer 0, so HTTP promotion is
  blocked pending a focused parity probe.

Detailed report:

```text
TEMP_STATUS_REPORT_423.md
```
