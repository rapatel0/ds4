# Sprint 589 - MTP layer-43 same-point body diagnostics

Date: 2026-05-30

## Why This Sprint Exists

Sprint 588 forced the unpromoted MTP layer to eager mode. That removed the
CUDA graph-capture host-copy failure and made it possible to collect
same-logical-point body diagnostics without adding permanent debug surface.

Acceptance is still `0/71`, so the next step is not another broad harness run.
It is a narrow MTP body localization pass: confirm whether layer-43 attention
state, raw-window output, attention output projection, and post-attention FFN
inputs are numerically sane at the exact draft point.

## Scope

1. Use temporary instrumentation only. Do not add permanent flags or smokes.
2. Force existing layer-43 diagnostics in the MTP-only `mtp_opt` path:
   - attention saturation state after raw state update;
   - attention output projection stats;
   - post-attention FFN input stats.
3. Build and run the existing deterministic acceptance harness once to collect
   those logs.
4. Remove temporary instrumentation unless it turns into a production fix.

## Non-Goals

- Do not change MTP token conditioning or position semantics.
- Do not implement speculative verification.
- Do not run a full performance A/B; this is correctness localization.

## Definition of Done

- Pod build passes for the temporary diagnostic build.
- Diagnostic logs are collected at layer 43 without graph-capture failure.
- The first newly suspicious stage, or the absence of obvious numeric failure,
  is recorded.
- Temporary instrumentation is removed from committed production code.

## Execution Result

Temporary diagnostics were added only to the MTP path and built on the pod:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

The existing deterministic acceptance harness still showed no accepted draft:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
```

Layer-43 diagnostics now run without graph-capture failure. Attention
projection, raw-state update, raw-window read, attention output projection, and
post-attention FFN input are finite and active. Representative values:

```text
tp_ep_true_attention_saturation_state layer 43 raw_row 0 q_heads_post_rope_bad 0 kv_post_rope_bad 0 raw_swa_row_bad 0 PASS
tp_ep_true_attention_output_projection layer 43 heads_bad 0 out_a_bad 0 out_b_bad 0 PASS
tp_ep_post_attention_ffn_input layer 43 post_bad 0 ffn_norm_bad 0 route_inv_scale_bad 0 PASS
```

A separate White House prompt smoke compared MTP-off control vs MTP-on with
temporary activation dumps:

```text
prompt: The address of the White House is:
temperature=0 top_p=1 max_tokens=32
control tokens == mtp tokens
MTP_ACCEPT_PAIRS 31 same 0 next 0
```

The served token stream is unchanged by MTP-on, so the snapshot/restore around
the MTP draft is correct. The in-path main output head before and after MTP
restore matched token, logit, and checksum for sampled steps.

Actual activation slices were dumped for the first slot across all ranks:

```text
layer42_final_hc_before_mtp: large finite layer-42 HC values
mtp_prologue_output: finite, nonzero MTP prologue activations
mtp_layer43_final_hc: finite, nonzero transformed MTP body activations
restored_layer42_final_hc: byte-identical to layer42_final_hc_before_mtp
```

The MTP raw-window progression is also not stuck. Across generated tokens,
layer 43 advanced position/raw row and valid rows as expected:

```text
position 0 raw_row 0 mtp_raw_valid_rows 1
position 1 raw_row 1 mtp_raw_valid_rows 2
position 2 raw_row 2 mtp_raw_valid_rows 3
...
```

Conclusion: the remaining 0% acceptance is not caused by a dead prologue,
stale input token, stuck raw cache, graph-capture path, or main-state clobber.
The next localization target should be a same-activation comparison of the MTP
head and/or layer-43 body math against a reference path, not more serving
parity runs. NVIDIA tools such as compute-sanitizer can still be useful for
invalid memory/race checks, but they will not provide a semantic numerical
oracle for this mismatch.
