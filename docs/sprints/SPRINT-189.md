# Sprint 189 - Single-Slot Attention Scratch Reuse

Date: 2026-05-22

## Objective

Remove per-layer-token attention tensor allocation churn from the single-slot
decode path by reusing the existing layer batch scratch arena when
`DS4_V100_SINGLE_SLOT_BATCH_SCRATCH=1`.

## Rationale

Sprint 187 showed filled-context direct replay is attention-dominated. Sprint
188 ruled out simple Q norm/RoPE launch fusion. The next safest attention-side
change is math-neutral: the single-slot attention body currently allocates and
frees eight temporary tensors every layer-token, while the batch path already
has reusable attention scratch. Reusing those buffers should not change model
math, cache visibility, or token selection.

## Scope

- Teach `execute_attention_output()` to use `cfg->batch_scratch` slot 0 when
  single-slot scratch is enabled.
- Keep default behavior unchanged until V100 evidence is positive.
- Validate with direct synthetic replay on the persistent Sprint 181 appliance
  pack.

## Non-Goals

- No online-attention promotion.
- No new attention softmax kernel.
- No KV format change.
- No TP/EP topology change.

## Definition of Done

- [x] V100 build passes.
- [x] Direct synthetic selected tokens match control for the explicit opt-in
      path.
- [x] A direct synthetic A/B is run with and without
      `DS4_V100_SINGLE_SLOT_BATCH_SCRATCH=1`.
- [x] Evidence is archived under `logs/from-cluster/`.
- [x] Sprint outcome and decision are recorded.
- [x] Vision is updated.
- [x] Changes are committed.

## Implementation

`execute_attention_output()` now can reuse slot-0 attention scratch when
`DS4_V100_SINGLE_SLOT_BATCH_SCRATCH=1`.

The first implementation passed oversized batch tensors for `q_a` and `kv_raw`
into single-row operations. It was fast and matched in one explicit opt-in A/B,
but the default-on experiment drifted token IDs. The patch was tightened to add
exact slot views for:

- `attn_q_a_view[slot]`
- `attn_kv_raw_view[slot]`

The selector remains default-off. `DS4_V100_SINGLE_SLOT_BATCH_SCRATCH=1`
enables the path, and the default path is unchanged.

## Evidence

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed.

Same-binary direct synthetic len-256 / ctx-262144:

| Mode | Prompt tok/s | Continuation tok/s | Output IDs |
|---|---:|---:|---|
| control | `12.534492` | `13.858673` | `3955, 361` |
| scratch opt-in | `16.162669` | `17.271478` | `3955, 361` |
| scratch opt-in, exact views | `16.256069` | `17.503002` | `3955, 361` |
| default-off rollback | `12.647768` | `13.907233` | `3955, 361` |
| default-on attempt | `16.319860` | `17.389163` | `1760, 582` |
| default-on with exact views | `16.330985` | `17.418521` | `1760, 582` |

Same-binary direct synthetic len-1024 / ctx-262144:

| Mode | Prompt tok/s | Continuation tok/s | Output IDs |
|---|---:|---:|---|
| control | `15.305026` | `15.228124` | `926, 926` |
| scratch opt-in | `16.901086` | `16.923312` | `926, 926` |

Evidence:

```text
logs/from-cluster/sprint189-single-slot-attn-scratch/
```

## Decision

Keep the single-slot attention scratch path default-off for now.

The explicit opt-in path is a real performance signal:

- len-256 continuation improved from `13.858673` to `17.503002` tok/s with
  matching IDs.
- len-1024 continuation improved from `15.228124` to `16.923312` tok/s with
  matching IDs.

However, attempts to promote the same idea as the no-env default produced
different first output IDs on len-256 (`1760, 582` instead of `3955, 361`),
even after tightening `q_a` and `kv_raw` to exact slot views. Because the
project prioritizes preserving source-model behavior, the path is not promoted
until that default-on discrepancy is understood.

Next work should either isolate why explicit opt-in and default-on selection
produce different IDs, or build a smaller single-slot scratch arena that avoids
the existing max-slot batch scratch machinery.
