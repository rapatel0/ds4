# Sprint 592 - MTP layer-43 dense projection oracle

Date: 2026-05-30

## Why This Sprint Exists

Sprint 590 cleared the MTP output head. Sprint 591 rejected the layer-43-only
inverse attention-head RoPE candidate. The remaining acceptance blocker is
still below the output head, and the next cheapest check is whether the MTP
layer-43 dense projections themselves are packed/oriented correctly.

The existing `run_dense_compute_gate` already compares the F8 dense kernel
against a CPU F8 dot-product oracle on synthetic inputs. This sprint applies
that existing diagnostic to the MTP layer only, without adding a permanent
flag or smoke.

## Scope

1. Temporarily enable `dense_compute_all_f8` for `mtp_opt` in the layer-43
   draft path.
2. Build on the pod and run the deterministic MTP acceptance harness.
3. Inspect all `dense_compute_tensor` rows for layer-43 MTP tensors.
4. If an oracle mismatch appears, fix the corresponding pack/layout bug.
5. If all dense tensors pass, remove the temporary instrumentation and move the
   next sprint to actual activation-boundary localization.

## Non-Goals

- Do not change token semantics.
- Do not add a permanent CLI flag or smoke.
- Do not run throughput A/B.
- Do not treat synthetic dense-oracle success as proof the full MTP body is
  correct; it only clears pack/orientation for the sampled dense rows.

## Definition of Done

- Temporary dense-oracle build passes on the pod.
- Deterministic harness runs and emits layer-43 dense-oracle rows.
- The result records whether any MTP dense tensor has repeat/oracle failures.
- Temporary instrumentation is removed before commit unless it becomes a real
  production fix.

## Result

Status: COMPLETE. The dense-oracle run exposed one real diagnostic-path bug,
but did not explain the zero-acceptance MTP draft.

The first build succeeded, but the acceptance harness produced no draft pairs
because `run_layer(43)` switched to the MTP contract while the diagnostic dense
compute path still read descriptors from the main `pack_dir`. The server log
failed on `blk.43.attn_kv_latent.weight`, proving the MTP layer inherited the
wrong pack root for descriptor-backed diagnostics.

Durable fix:

- `engine/token_major_loop.cu`: set `mtp_opt.pack_dir` to
  `opt.mtp_pack_dir ?: opt.pack_dir` before invoking `run_layer(43)` in the
  serving draft path.
- `engine/appliance_runtime.cu`: apply the same override in the appliance
  layer-43 scaffold.

With that fix and the temporary `dense_compute_all_f8` hook enabled, the pod
harness emitted layer-43 dense-oracle rows for all MTP F8 dense tensors:

- `blk.43.attn_kv_latent.weight`
- `blk.43.attn_output_a.weight`
- `blk.43.attn_output_b.weight`
- `blk.43.attn_q_a.weight`
- `blk.43.attn_q_b.weight`
- `blk.43.e_proj.weight`
- `blk.43.ffn_down_shexp.weight`
- `blk.43.ffn_gate_shexp.weight`
- `blk.43.ffn_up_shexp.weight`
- `blk.43.h_proj.weight`

Every row passed. The observed oracle error was float noise
(`oracle_max_abs <= 0.000000030`, `oracle_bad = 0`, `repeat_bad = 0`). This
rules out the obvious MTP dense F8 pack/orientation failure class.

The deterministic acceptance harness still rejected the draft:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12] [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]
```

The temporary dense-oracle hook was removed after the diagnostic run. The clean
pod build then passed:

```text
BUILD_EXIT=0
make appliance/ds4-v100-tp-ep-appliance
```

Next sprint should localize the activation/state boundary below the dense
matmuls: MTP body control tensors, HC mix/control application, raw-SWA state,
or the post-attention/FFN activation sequence. Do not build the K-wide verifier
or run throughput A/B until draft acceptance is nonzero.
