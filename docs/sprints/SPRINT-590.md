# Sprint 590 - MTP output-head same-activation reference

Date: 2026-05-30

## Why This Sprint Exists

Sprint 589 proved that the MTP path is connected, finite, input-responsive,
and non-clobbering: MTP-on serves the same main tokens as MTP-off, layer 43
advances its raw-window rows, and actual activation slices move through the
prologue and body. Acceptance is still `0/71`, so another serving parity run
would not localize the defect.

The next smallest numerical boundary is the MTP output head. It consumes the
layer-43 final HC activation and produces the draft token. If the output-head
preparation is wrong, layer-43 body work is a distraction. If it matches a
same-activation reference, the defect moves below the head into the layer-43
body math.

## Scope

1. Use temporary instrumentation only; do not add permanent flags, smokes, or
   root temp files.
2. On an actual MTP draft call, copy slot-0 layer-43 final HC shards from all
   ranks and compare the GPU output-head preparation against a CPU reference:
   - HC flat RMS + `blk.43.hc_head_fn` preactivation;
   - `blk.43.hc_head_scale/base` sigmoid weights;
   - weighted HC sum;
   - `blk.43.norm.weight` final embedding RMS norm per rank.
3. Compare the CPU reference with the GPU buffers produced by
   `run_shared_output_head_from_rank_hc`: `d_head_pre_rank`,
   `d_head_weights_rank`, and `d_embd_norm_shard`.
4. Do not CPU-reference the full vocab projection in this sprint; the shared LM
   projection is already exercised by the main head and would require a large
   full-vocab host matvec.

## Non-Goals

- Do not change MTP token-conditioning semantics.
- Do not implement the K-wide speculative verifier.
- Do not run a performance A/B.
- Do not make the diagnostic a committed runtime option.

## Definition of Done

- Pod build passes for the temporary diagnostic build.
- The deterministic MTP acceptance harness or an equivalent deterministic HTTP
  run emits same-activation output-head reference comparisons.
- The result either identifies an output-head mismatch to fix or clears the
  output-head preparation and moves the next sprint to layer-43 body
  localization.
- Temporary instrumentation is removed from committed production code.

## Execution Result

Temporary same-activation instrumentation was added to the MTP draft-head path
and built on the pod:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

The first diagnostic run found a real output-head preparation mismatch. The
CPU reference loaded `blk.43.hc_head_fn/base/scale/norm.weight`, copied the
actual slot-0 layer-43 final HC shards from all ranks, and compared against
the GPU buffers produced by `run_shared_output_head_from_rank_hc`:

```text
tp_ep_mtp_head_cpu_ref step 0 hc_max 22.7986088 mix_cpu0 3.56024885 mix_gpu0 -0.78537488 mix_max_abs 9.35504174 weight_cpu0 0.988725543 weight_gpu0 0.837492406 weight_max_abs 0.209268968 norm_max_abs 0.00682156443 norm_rms 0.00220216954 PASS
tp_ep_mtp_head_cpu_ref step 0 hc_max 56.8990555 mix_cpu0 4.29839706 mix_gpu0 1.05443728 mix_max_abs 9.50455928 weight_cpu0 0.989110887 weight_gpu0 0.938780189 weight_max_abs 0.161800847 norm_max_abs 0.00893342137 norm_rms 0.00211425303 PASS
```

Root cause: the rank-local `hc_head_fn` slice treated each rank as owning one
contiguous `4 * shard_cols` span of the flattened HC vector. The real HC
layout is row-major `[4][4096]`; each rank owns `512` columns inside every HC
row. The fixed slice is:

```text
global_c = hc_row * 4096 + rank * 512 + local_col
```

The fix was applied to both `open_shared_output_head` and
`load_mtp_output_head`, because the same rank-local head-preparation code is
shared by the main and MTP output heads.

After the fix, the same diagnostic collapsed to float-level agreement:

```text
tp_ep_mtp_head_cpu_ref step 0 hc_max 22.7986088 mix_cpu0 3.56024885 mix_gpu0 3.56024885 mix_max_abs 4.76837158e-07 weight_cpu0 0.988725543 weight_gpu0 0.988725543 weight_max_abs 5.58793545e-09 norm_max_abs 1.34300483e-07 norm_rms 1.75647568e-08 PASS
tp_ep_mtp_head_cpu_ref step 0 hc_max 56.8990555 mix_cpu0 4.29839706 mix_gpu0 4.29839706 mix_max_abs 0 weight_cpu0 0.989110887 weight_gpu0 0.989110887 weight_max_abs 2.79396772e-09 norm_max_abs 1.11819677e-07 norm_rms 2.15263057e-08 PASS
```

The deterministic MTP acceptance harness still rejects the draft:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12] [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]
```

The temporary instrumentation was removed and the clean source was recopied to
the pod workspace. The clean build and acceptance rerun passed operationally
and emitted no `tp_ep_mtp_head_cpu_ref` lines:

```text
BUILD_EXIT=0
ACCEPT_EXIT=0
REF_COUNT 0
pairs 71 same_index_match 0 (0.00) next_index_match 0
```

Conclusion: the output-head preparation had a real rank-local HC-control
slicing bug and is now reference-clean. That bug did not resolve MTP draft
acceptance, so the remaining blocker is below the head, in layer-43 body math
or body state. Next sprint should apply the same same-activation reference
method to layer-43 body boundaries, starting with attention projection/raw-SWA
then post-attention FFN output.
