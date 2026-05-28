# TEMP Status Report 420

Date: 2026-05-27

## Sprint

Sprint 416: Rank-Local Attention Projection Input

## Objective

Test whether true DS4 attention projection input can avoid graph-captured
device-0-to-rank full-hidden copies by doing attention RMS norm on each TP rank
and filling `attn_q_a` / `attn_kv_latent` from the rank-local normalized
hidden buffer.

This remains TP/EP-only. No PP/layer-split variants were tested.

## Implementation

Added an opt-in gate:

```text
--true-ds4-attention-projection-rank-local-input-gate
```

When enabled:

- `attn_norm.weight` is replicated to each rank.
- Device 0 still computes canonical `hc->d_attn_normed` for downstream
  semantic consumers.
- Each rank runs the same RMS norm on its local `r.d_current_full`.
- `attn_q_a` and `attn_kv_latent` half inputs are filled from that local
  normalized buffer.
- The projection path avoids the graph-safe full-hidden copy from device 0 to
  every rank.

## Build

V100 build passed:

```text
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Resident Layer 2 A/B

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/resident-layer2-baseline/
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/resident-layer2-ranklocal-rebuilt/
```

Results:

| Mode | Checksum | Capture | Replay ms | Decode ms/step | Slot-step tok/s | Nodes |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8290057485 | pass | 9.905152 | 2.476288 | 3230.641889 | 789 |
| rank-local | 8290057485 | pass | 9.219072 | 2.304768 | 3471.065072 | 789 |

Resident result: positive. Same checksum, same graph node count, and about
6.9% faster replay for layer 2.

## All-Layer Direct Decode A/B

The first all-layer rank-local run at scratch `512 MiB` passed:

```text
artifact=/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/all-layer-ranklocal-slot8-tokens4/
aggregate_generated_tok_s_decode=93.980440
aggregate_continuation_tok_s_decode=107.202645
checksum=4335215310
capture_succeeded=43/43
replay_succeeded=172/172
PASS
```

For a clean same-binary A/B after rebuild, the all-layer control at scratch
`512 MiB` OOMed during shared expert pack allocation. The same command shape
passed at scratch `256 MiB`, so the clean A/B used scratch `256 MiB` for both
control and candidate.

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/ab-clean-baseline-slot8-tokens4-scratch256/
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/ab-clean-ranklocal-slot8-tokens4-scratch256/
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

Results:

| Mode | Generated decode tok/s | Continuation decode tok/s | Checksum | Replay | Capture |
|---|---:|---:|---:|---:|---:|
| baseline | 84.072506 | 94.326524 | 4335215310 | 172/172 | 43/43 |
| rank-local | 92.702737 | 105.428529 | 4335215310 | 172/172 | 43/43 |

Delta:

```text
generated decode:     +10.26%
continuation decode:  +11.77%
checksum:             unchanged
graph capture/replay: unchanged success
```

## Decision

Rank-local attention projection input is a positive direct-decode candidate.

Do not yet make it the production serving default solely from this sprint. The
next promotion step is an HTTP serving A/B at the practical long-context tier
with response parity/readiness checks. The direct-decode evidence is strong
enough to keep this gate as the next serving candidate.

## Additional Finding

Current all-layer shared expert residency is now very tight. The successful
rank-local scratch-512 run reported:

```text
tp_ep_all_layer_expert_bindings_shared bytes=147169738752 bytes_per_gpu=18396217344
```

A same-binary control run with scratch `512 MiB` OOMed at expert pack
allocation:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:8654: out of memory
```

This is separate from the rank-local attention change because it happens
before the attention projection path is reached. The next memory/performance
sprint should either reduce full expert residency, make expert residency lazy
or staged, or make the planner account for this higher packed expert footprint.

## Next Bottleneck Direction

Use rank-local input as the next pattern to test in serving:

- promote through HTTP A/B only if response parity and readiness pass
- keep direct remote-source fill rejected
- preserve local staging, but feed downstream dense/attention consumers from
  local rank buffers
- reduce shared expert residency pressure before returning to larger scratch
  and 32-slot target serving tests
