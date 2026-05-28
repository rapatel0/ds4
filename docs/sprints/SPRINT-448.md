# Sprint 448: HC-Current Full-Buffer Parity

## Objective

Stay TP/EP only and determine whether the rank-local attention input divergence
comes from inconsistent per-rank `RankState::d_current_full` buffers after
HC-current NCCL allgather and slot-major conversion.

Sprint 447 proved the attention rank-local token divergence happens before the
Q/KV dense projections. The next narrow check is the full hidden buffer that
rank-local attention normalizes on each GPU.

## Implementation Plan

1. Add a default-off diagnostic gate:

   ```text
   --tp-hc-current-full-parity-gate
   ```

2. After HC-current NCCL allgather and
   `rank_major_current_shards_to_slot_major_kernel`, compare every rank's
   `d_current_full` against rank 0.
3. Use host copies for this diagnostic so the result is not affected by
   peer-memory reads from inside a GPU kernel.
4. Run the direct all-layer TP/EP audit at `8` slots / `256K` with the attention
   input parity gate still enabled.

## Validation

- Local diff check.
- Remote sm_70 build.
- V100 direct audit emits `tp_ep_hc_current_full_rank_diff`.
- No stale DS4 GPU process remains after the run.

## Decision Rule

- If `d_current_full` differs across ranks, fix HC-current NCCL allgather or
  rank-major-to-slot-major conversion.
- If `d_current_full` matches across ranks, move the next fix to per-rank
  RMSNorm/weight residency or the attention half-input parity reference path.

## Execution Notes

Implemented the default-off diagnostic gate:

```text
--tp-hc-current-full-parity-gate
```

The gate logs `tp_ep_hc_current_full_rank_diff` immediately after the
HC-current NCCL allgather and rank-major-to-slot-major conversion, using host
copies from each rank so the diagnostic does not depend on peer reads inside a
GPU kernel.

The direct all-layer run also kept
`--true-ds4-attention-projection-input-parity-gate` enabled so the same artifact
checks both the gathered full-hidden buffer and the actual half inputs consumed
by the attention Q/KV projections.

## Outcome

Artifacts:

- Direct HC-current full parity audit:
  `/localpool/ds4/workspace/logs/s448-hc-current-full-parity-direct`
- Rebuilt HTTP attention rank-local A/B:
  `/localpool/ds4/workspace/logs/s447-http-attn-ranklocal-broadcast`

Direct audit summary:

| Line family | Lines | Bad lines | Result |
|---|---:|---:|---|
| `tp_ep_hc_current_full_rank_diff` | 344 | 0 | pass |
| `tp_ep_attention_projection_input_diff` | 688 | 0 | pass |

The direct audit completed with `rc=0`. Every rank's
`RankState::d_current_full` matched rank 0 on every layer at the reduced
`8` slot / `256K` shape, and the actual half inputs for both
`attn_q_a_input` and `attn_kv_latent_input` matched the normalized reference.

The rebuilt same-binary HTTP A/B then confirmed correctness for the
attention-rank-local input candidate:

| Leg | Server generated decode tok/s | Server continuation decode tok/s | First token | Response parity |
|---|---:|---:|---:|---:|
| Control | 20.225169 | 20.265472 | 71302 | 4/4 |
| Rank-local attention input | 18.383328 | 18.937739 | 71302 | 4/4 |

The HTTP wrapper still has a parent-process handoff issue after each leg: the
profile leg finishes, but the A/B parent sometimes needs an explicit signal
before launching the next leg. The response artifacts are complete and valid,
but the harness handoff should be fixed separately.

## Decision

The HC-current allgather and slot-major full-hidden conversion are correct at
this tested shape. The earlier attention half-input mismatch was eliminated by
the source/buffer fix in the rebuilt binary.

Do not promote `--true-ds4-attention-projection-rank-local-input-gate` yet:
correctness is now clean, but this small HTTP run regressed server generated
decode from `20.225169` to `18.383328` tok/s. Keep the gate opt-in until it is
combined with the next rank-major/launch-reduction changes and tested at a
larger serving shape.
