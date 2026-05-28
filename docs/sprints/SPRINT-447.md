# Sprint 447: Attention Projection Current-Source Fix

## Objective

Stay TP/EP only and fix the attention projection input source identified by
Sprint 446.

Sprint 446 showed that `--true-ds4-attention-projection-rank-local-input-gate`
changes tokens by itself. Code inspection found that when HC-current NCCL or
peer gather is active, the fresh full hidden state lives in per-rank
`RankState::d_current_full`, while the attention projection prefix still
normalizes `SharedHcControls::d_current_full`. That can compare a fresh
rank-local candidate against a stale device-0 control source.

## Implementation Plan

1. In `run_true_ds4_attention_projection_prefix`, select the current-full source
   from `ranks[0].d_current_full` when HC-current peer/NCCL gather is active.
2. Keep the fallback as `hc->d_current_full` for non-rank-local/non-gather
   modes.
3. Use that selected source for the device-0 attention RMSNorm and diagnostic
   stats.
4. Rebuild the V100 TP/EP server binary.
5. Rerun the attention-only A/B at the Sprint 446 reduced isolation shape.

## Validation

- Local syntax/diff checks.
- Remote sm_70 build.
- V100 attention-only HTTP A/B with readiness and response parity.
- No stale DS4 GPU processes after the run.

## Decision Rule

If attention-only parity becomes clean, the previous token change was a stale
control-source bug. Then rerun the combined rank-major candidate before
promotion.

If attention-only still fails parity, add direct pre-dense input-buffer parity
for `attn_q_a` and `attn_kv_latent`.

## Execution Notes

The current-source fix changed both control and candidate output tokens, proving
the stale-source path was real, but attention-only parity still failed. Added a
default-off diagnostic gate:

```text
--true-ds4-attention-projection-input-parity-gate
```

It logs `tp_ep_attention_projection_input_diff` before the attention Q/KV dense
projections and compares the actual half inputs for `attn_q_a` and
`attn_kv_latent` against the device-0 normalized attention input.

## Outcome

Artifacts:

- HTTP A/B after current-source fix:
  `/localpool/ds4/workspace/logs/s447-attn-source-fix-attn-only-ab`
- Direct input parity audit:
  `/localpool/ds4/workspace/logs/s447-attn-input-parity-direct`

The current-source fix changed both control and candidate outputs, confirming
that the stale `hc->d_current_full` source was real:

| Leg | First token | Server decode tok/s | Readiness |
|---|---:|---:|---|
| Control after fix | 71302 | 20.326755 | pass |
| Rank-local candidate after fix | 63930 | 20.594887 | pass |

However, attention-only parity still failed `0/4`.

The direct input parity audit then proved the remaining divergence happens
before the attention Q/KV dense projections:

```text
tp_ep_attention_projection_input_diff
```

Summary:

| Family | Lines | Bad lines | Mismatches |
|---|---:|---:|---:|
| `attn_q_a_input` | 344 | 10 | 325087 |
| `attn_kv_latent_input` | 344 | 10 | 325087 |

First mismatches:

```text
layer 0 rank 1 attn_q_a_input        mismatches 32535 max_abs 0.135314941
layer 0 rank 1 attn_kv_latent_input  mismatches 32535 max_abs 0.135314941
layer 0 rank 6 attn_q_a_input        mismatches 32535 max_abs 0.135314941
layer 0 rank 6 attn_kv_latent_input  mismatches 32535 max_abs 0.135314941
layer 0 rank 7 attn_q_a_input        mismatches 32535 max_abs 0.135314941
layer 0 rank 7 attn_kv_latent_input  mismatches 32535 max_abs 0.135314941
```

The direct audit completed all 43 layers with `rc=0`, but the parity mismatch is
material and explains the HTTP token divergence.

## Decision

Do not promote the attention rank-local input gate.

The stale device-0 current source is fixed, but the rank-local attention input
still consumes inconsistent per-rank full-hidden buffers. The next sprint should
audit `RankState::d_current_full` immediately after HC-current NCCL allgather
and `rank_major_current_shards_to_slot_major_kernel`, comparing every rank's
slot-major full-hidden buffer against rank 0 before RMSNorm. Fix that
per-rank current consistency before returning to attention input A/Bs.
