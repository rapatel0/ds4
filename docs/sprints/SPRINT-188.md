# Sprint 188 - Fused Q Norm/RoPE Attention Probe

Date: 2026-05-22

## Objective

Attack the Sprint 187 filled-context attention bucket with a bounded
quality-preserving launch fusion: combine Q head RMS normalization and RoPE into
one CUDA dispatch behind an opt-in gate.

## Rationale

Sprint 187 repaired direct synthetic profile buckets and showed the
len-1024 / ctx-262144 filled-context path is dominated by the attention body:

- Attention bucket: `37779.266 ms` / `56.8%`
- FFN bucket: `21473.437 ms` / `32.3%`
- Handoff sum: `219.707 ms`

The attention bucket includes projection, Q/KV normalization, RoPE, cache
update, attention softmax, inverse RoPE, and output projection. The current
single-slot path already has an unused CUDA primitive that fuses head RMS norm
with RoPE. This is a small, local probe that avoids changing model layout,
cache layout, or attention visibility.

## Scope

- Expose `ds4_gpu_head_rms_norm_rope_tail_tensor()` in the public CUDA header.
- Add a DS4 layer helper that computes the same RoPE parameters as the existing
  helper and calls the fused primitive.
- Select the fused Q norm/RoPE path only when
  `DS4_V100_FUSED_Q_NORM_ROPE=1`.
- Apply it to direct single-slot attention and the existing batch attention
  path.
- Validate on the persistent Sprint 181 appliance pack.

## Non-Goals

- No online-attention promotion.
- No change to default behavior unless the evidence is clearly positive and
  selected-token behavior is acceptable.
- No KV format change.
- No TP/EP topology change.

## Definition of Done

- [x] V100 build passes.
- [x] Selected-token smoke passes with the opt-in enabled.
- [x] A direct synthetic filled-context A/B is run against the current default.
- [x] Evidence is archived under `logs/from-cluster/`.
- [x] Sprint outcome and decision are recorded.
- [x] Vision is updated.
- [x] Changes are committed.

## Implementation

Added `DS4_V100_FUSED_Q_NORM_ROPE=1`, default-off.

The opt-in path:

- Exposes `ds4_gpu_head_rms_norm_rope_tail_tensor()` in `ds4_gpu.h`.
- Adds `q_head_norm_rope_tail_layer_tensor()` so DS4 compressed/non-compressed
  RoPE parameters stay centralized with the existing helper.
- Selects the fused Q head-RMS + RoPE primitive in both:
  - `execute_attention_output()`
  - `execute_attention_output_batch()`

No default behavior changed.

## Evidence

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed.

Same-binary direct synthetic A/B:

```text
--appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181
--synthetic-prompt-token 926
--synthetic-prompt-len 256
--ctx 262144
--tokens 2
```

| Mode | Prompt tok/s | Continuation tok/s | Output IDs |
|---|---:|---:|---|
| control | `12.698799` | `14.068688` | `3955, 361` |
| fused Q norm/RoPE | `12.399597` | `12.024425` | `3955, 361` |

Evidence:

```text
logs/from-cluster/sprint188-fused-q-norm-rope/
```

## Decision

Reject for production default and keep the path diagnostic-only.

The token IDs matched for this bounded direct replay, but the fused path
regressed continuation throughput by roughly `14.5%`. The likely reason is that
the fused head-RMS/RoPE kernel reduces one dispatch but does more per-head work
than the existing split path, so it loses in the actual V100 attention shape.

This sprint is still useful because it rules out a simple launch-count fusion
inside the Q normalization/RoPE section. The next attention/KV sprint should
target reusable single-slot attention scratch or finer subprofiling of
projection/cache/softmax/output buckets, not this fused Q path.
