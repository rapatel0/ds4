# TEMP Status Report 422

Date: 2026-05-27

## Focus

Pivot the positive rank-local attention projection gate toward the stricter
rank-major strategy: consume the NCCL allgather rank-major hidden buffer
directly instead of normalizing a slot-major full tensor on each rank.

This stayed TP/EP-only. No PP/layer-split work.

## Implementation

The existing opt-in gate remains:

```text
--true-ds4-attention-projection-rank-local-input-gate
```

When HC-current NCCL allgather is active, the gate now uses a fused kernel:

```text
fill_two_hidden_inputs_half_from_rank_major_norm_kernel
```

The kernel reads rank-major current hidden:

```text
[rank][slot][hidden / 8]
```

and directly writes the two half inputs used by attention projection:

```text
attn_q_a.d_x_half
attn_kv_latent.d_x_half
```

It fuses:

- full-row RMS norm over the logical hidden row
- `attn_norm.weight` application
- rank-major to slot-major addressing
- F32 to F16 conversion
- duplicate fill of the two projection inputs

The old per-rank slot-major path remains as fallback when NCCL rank-major
input is unavailable.

## Validation

V100 build passed:

```text
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Resident Layer 2

Artifact:

```text
/localpool/ds4/workspace/logs/sprint422-rankmajor-attn-proj/resident-layer2-rankmajor-clean/
```

Result:

| Metric | Value |
|---|---:|
| checksum | 8290057485 |
| capture | pass |
| replay | pass |
| graph nodes | 773 |
| replay ms | 9.169920 |
| decode ms/step | 2.292480 |
| slot-step tok/s | 3489.670587 |

Comparison to the previous Sprint 416 resident data:

| Mode | Replay ms | Decode ms/step | Slot-step tok/s | Nodes |
|---|---:|---:|---:|---:|
| baseline | 9.905152 | 2.476288 | 3230.641889 | 789 |
| rank-local slot-major | 9.219072 | 2.304768 | 3471.065072 | 789 |
| rank-major fused | 9.169920 | 2.292480 | 3489.670587 | 773 |

## All-Layer Direct Decode

Artifact:

```text
/localpool/ds4/workspace/logs/sprint422-rankmajor-attn-proj/full-rankmajor-slot8-tokens4-scratch256/
```

Shape:

```text
slots=8
ctx=262144
decode_steps=4
tp-runtime-scratch=256 MiB
defer-nccl-init=on
hc-current-nccl=on
persistent graph replay=on
```

Result:

| Metric | Value |
|---|---:|
| generated decode tok/s | 93.586972 |
| continuation decode tok/s | 106.584476 |
| checksum | 4335215310 |
| capture | 43/43 |
| replay | 172/172 |
| capture nodes | 111492 |

Comparison to Sprint 416 clean direct A/B:

| Mode | Generated decode tok/s | Continuation decode tok/s | Checksum |
|---|---:|---:|---:|
| baseline | 84.072506 | 94.326524 | 4335215310 |
| rank-local slot-major | 92.702737 | 105.428529 | 4335215310 |
| rank-major fused | 93.586972 | 106.584476 | 4335215310 |

## Decision

Rank-major consumption is the correct direction. This change is a small but
clean improvement over the previous rank-local slot-major gate and it reduces
captured graph nodes for the resident layer.

Do not spend more work on PP/layer variants or device-0 staging variants. The
next sprint should continue converting full-hidden consumers from:

```text
gather to device 0 -> compute -> redistribute
```

to:

```text
rank-major consumer, or sharded reduction plus tiny collective
```

## Operational Note

Two stale host-launched slot28 HTTP retry containers were found holding GPU
memory while not serving. They were terminated before the clean resident/full
validation runs. Future cluster runs should stay inside one controlled pod or
write a visible lock file before launching host `ctr` benchmarks.

## Next

1. Convert FFN/router RMS norm to a sharded/rank-major path.
2. Stop materializing slot-major `r.d_current_full` when every downstream
   consumer in that layer-step can use rank-major or shard-local input.
3. Re-run HTTP selected-token and chat at 8 slots, then retry the 28-slot tier.
4. Address expert-residency headroom before attempting 32-slot production
   default at 256K.
