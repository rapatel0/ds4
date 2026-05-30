# Sprint 591 - MTP layer-43 attention-head semantic probe

Date: 2026-05-30

## Why This Sprint Exists

Sprint 590 fixed and cleared the MTP output-head preparation boundary, but the
draft still has `0/71` acceptance. The next blocker is therefore below the
head, inside layer-43 body math or body state.

The first concrete semantic discrepancy is in the attention path. Upstream
`research/ds4/ds4.c` computes raw-SWA attention heads and then applies inverse
RoPE to the heads before the grouped attention output projection:

```text
layer_attention_rows_one(...)
rope_tail_layer_inplace(heads, ..., inverse=true)
layer_grouped_out_one(...)
```

The TP/EP path currently applies RoPE to Q and KV before raw-SWA storage/read,
but `run_true_ds4_attention_raw_window` feeds `r.d_attn_heads` directly into
`run_true_ds4_attention_output_projection`. This sprint tests the missing
inverse-head-RoPE hypothesis on the MTP layer first, because it is small,
same-stage, and does not require a full CPU oracle.

## Scope

1. Use a temporary layer-43-only candidate that applies inverse RoPE to
   `r.d_attn_heads` after raw-SWA attention and before attention output
   projection.
2. Build on the pod and run the existing deterministic MTP acceptance harness.
3. Inspect whether draft tokens, logits, or acceptance move materially.
4. If acceptance improves or the diagnostic clearly fixes a semantic mismatch,
   keep the minimal production fix and broaden validation as needed.
5. If it does not move acceptance, remove the candidate and record the result.

## Non-Goals

- Do not add a permanent runtime flag.
- Do not implement K-wide verification.
- Do not change the main serving path unless the layer-43 probe produces a
  clear correctness signal that justifies broadening the fix.
- Do not run throughput A/B; this is correctness localization.

## Definition of Done

- Pod build passes for the temporary candidate.
- The deterministic acceptance harness runs to completion.
- The result records whether inverse-head-RoPE changes MTP acceptance or draft
  behavior.
- Temporary candidate code is either promoted as a justified fix or removed
  before commit.

## Execution Result

Temporary code applied inverse RoPE to `r.d_attn_heads` only for layer 43 after
`run_true_ds4_attention_raw_window` and before attention output projection.
The pod build passed:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

The deterministic acceptance harness completed, and the candidate changed the
MTP draft tokens, proving the probe was active:

```text
baseline draft[:12]  [112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]
candidate draft[:12] [18560, 5743, 80610, 84941, 84941, 84941, 82318, 124211, 68268, 123327, 30701, 67132]
```

Acceptance did not improve:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12] [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
```

Decision: reject the layer-43-only inverse-head-RoPE candidate as the immediate
MTP acceptance fix. The temporary code was removed before commit. This result
does not prove the broader TP/EP attention path is semantically perfect; it
only says that adding inverse head RoPE to the MTP layer alone does not move
acceptance off zero. The next localization step should validate layer-43
dense/projection correctness and HC body controls on actual or synthetic
inputs, rather than chasing this candidate further.

After removing the temporary code, the clean pod workspace was rebuilt:

```text
BUILD_EXIT=0
```
