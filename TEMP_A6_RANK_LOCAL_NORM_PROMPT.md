# A6 — Fix rank-local attention RMS-norm (the next big bottleneck)

After A2/A3 land, the attention pre-EP staging (~240 ms / ~27% of decode time,
excluding `pre_ep_hc_current` which A2 attacks) is the dominant GPU0-centralized
chunk left in HC-current. A6 is the gateway: an existing implementation exists
(`--attention-projection-rank-local-input`) but it fails parity catastrophically
(s480: **1/32 selected-token agreement**, max rel-err **0.088**).

The job of this sprint is to **diagnose and fix the existing A6 path** so it
matches the relaxed-gate quality bar (agreement ≥ 0.99 on both selected-token
and generated-sequence), the same way A2/A3 do.

## The real picture (verified from the code, 2026-05-28)

**A complete rank-major norm implementation already exists in the tree — it's
hardcoded off.** Three findings:

1. **`fill_two_hidden_inputs_half_from_rank_major_norm_kernel`** (lines
   2220–2278) is a correct, complete norm-and-fill kernel reading rank-major
   `[rank, slot, shard_col]` layout and producing Q/KV `__half` inputs. The
   math (max-abs / `Σ(v/max_abs)²` / scale / `v·scale·weight[col]`) is
   bit-identical to the centralized `rms_norm_weight_rows_stable_kernel` on
   slot-major data.
2. **The dispatch site is dead** because of one hardcoded line. At line 13448:
   ```cpp
   const bool rank_major_input = false;   // hardcoded in this function only
   ```
   The same variable is properly gated in the routed-FFN function (line 16170)
   — only the attention-projection function hardcodes it false. The check at
   13509 (`else if (rank_major_input && r.d_current_full_rank_major)`) is
   therefore never taken.
3. **The s480 1/32 result is from a different sub-branch** — the `else` at
   13517 — being exercised under a gate combination that breaks data
   consistency. It is not evidence that the rank-major path is broken.

## Why this is the actual A6 win

When the rank-major path runs:
- The `tp_hc_current_input_nccl_allgather_gate` allgather populates
  `r.d_current_full_rank_major` equivalently on every rank (one NCCL
  collective, ~512 KB).
- Each rank runs the norm kernel **locally, in parallel** on its identical
  rank-major buffer.
- **No GPU0-serial norm step. No `ncclBroadcast` of normed.** Eight ranks
  redundantly compute the same result concurrently; the broadcast disappears.

The arithmetic is **bit-identical to control** (same values summed in the same
order, same scale). This is **not** a Pattern A tolerance change — it should
pass the strict bit-exact gate. It's a transport + parallelization change that
happens to produce the same fp32 result.

## The correct math identity (the fix is mechanical)

Same pattern as A2. Centralized RMS-norm:

```
scale[slots] = rsqrt( mean_over_4096( x_full[slot, :]² ) + eps )
normed_full[slot, h] = x_full[slot, h] · scale[slot] · weight[h]
```

Rank-local equivalent, **provably equal up to fp32 reduction order**:

```
# each rank (8 of them), holding x_shard[slots, 512]:
partial_sumsq[slots] = Σ_local x_shard²              # rank-local
global_sumsq[slots]  = ncclAllReduce(partial_sumsq, ncclSum,
                                     r.compose_nccl) # KEY STEP — this is what's missing
scale[slots]         = rsqrt(global_sumsq / 4096 + eps)
normed_shard[slot, h_local] = x_shard[slot, h_local]
                            · scale[slot]
                            · weight_shard[h_local]   # weight sharded same as x
```

**Trap:** verify the norm `weight` (gamma) is sharded along the hidden axis so
each rank applies its 512-wide slice. If `weight` is replicated (all 4096 on
each rank), index it as `weight[rank*512 + h_local]`.

**Optional polish (apply *after* the correctness fix lands):** fp64 partial
sumsq accumulation, rounded to fp32 at the end, to tighten the drift number. Not
required to promote — gate is agreement, not rel-err.

## Implementation tasks (much smaller than the previous draft)

This is a **dispatch fix + buffer-lifetime fix**, not new kernel work.

1. **Un-hardcode the rank-major dispatch.** Line 13448, change
   `const bool rank_major_input = false;` to be driven by the prerequisite
   gate, e.g.
   ```cpp
   const bool rank_major_input =
       opt.tp_hc_current_input_nccl_allgather_gate;
   ```
   Or, cleaner: introduce a dedicated gate
   `--attn-projection-rank-major-norm-gate` so this function's behavior is
   decoupled from the HC-current allgather's other effects. Either way, the
   existing `rank_major_input && r.d_current_full_rank_major` runtime guard at
   line 13509 is the safety net.
2. **Audit buffer lifetime.** The flow at ~7495 currently does
   `ncclAllGather → r.d_current_full_rank_major`, then
   `rank_major_current_shards_to_slot_major_kernel` transposes into
   `r.d_current_full`. The rank-major buffer must remain populated and
   untouched until the attention-projection norm kernel consumes it. Either:
   - keep the rank-major buffer live across the transpose (it likely already
     is — it's a separate allocation), or
   - re-issue the allgather just for the attn-projection consumer if the
     transpose path freed/reused it.
3. **Verify the existing `else if (rank_major_input && r.d_current_full_rank_major)`
   branch (lines 13509–13516) is the only call site of the rank-major-norm
   kernel** and that no surrounding code assumes the broadcast sub-branch ran.
   In particular, verify nothing downstream reads `r.d_current_full_normed`
   when the rank-major branch is taken (the rank-major branch writes directly
   to the Q/KV `__half` inputs and bypasses `d_current_full_normed`).
4. **Promote under the strict bit-exact gate.** Math is identical to control;
   parity should be **256/256 selected-token** and rel-err should be **0**.
   If not, the dispatch is misrouted (the buffer wasn't populated, or another
   sub-branch is still running).

## What stays NOT to do

- **Do not "fix" the `else` at 13517–13533.** It's a separate broken sub-path
  with no perf upside even at 1.0 agreement (it replicates the centralized
  norm computation but doesn't avoid the broadcast). After the rank-major
  path is promoted, the `else` sub-branch and the buggy gate combination that
  reaches it can be deleted as part of the cleanup sprint.
- **Do not bundle Pattern B row-parallel Q/KV projection.** This sprint
  decentralizes the norm only. Pattern B is a separate sprint with FP8
  kernel re-tune cost.
- **Do not build the partial-`Σx²` + `ncclAllReduce` design** I described in
  the previous draft. It's unnecessary — the existing kernel produces the
  same result by parallel replicated computation on the rank-major buffer.
  The "redundant compute on 8 ranks" cost is negligible for a small norm
  kernel.

## Out of scope for this sprint

- A4b row-parallel attention Q/KV projection (option (b) above).
- The other attention-staging buckets (compressed KV, attention state /
  output, post-attention FFN input). Once the norm pattern works, these are
  each their own follow-on (same template).
- Fusion of HC-pre into the attention-projection prologue (the original meaning
  of "A6" in `SPIKE_B_STEERING.md` was fusion, not just rank-local norm).
- EP / MTP / graph capture.

## Gating

Per `TEMP_PARITY_POLICY.md` (relaxed):

- **PRIMARY**: selected-token agreement ≥ 0.99 AND generated-sequence
  agreement ≥ 0.99 on the reference shape (32 slots / 256K / 256 req / 64 tok)
  vs control (current default-on path, which is GPU0-norm + ncclBroadcast).
- **Advisory**: max selected-logit rel-err. Expect ~0.02–0.10 (A2/A3-class drift)
  once the bug is fixed. **Do not gate on rel-err.**
- **No regression**: `peer_copy_sys_bytes = 0` (transport is unchanged from
  the s479 baseline), decode tok/s at-or-above control.
- **No rerun if it passes**: one A/B, promote if agreement clears the bar.

## Expected magnitude

The norm itself is small (slots × 4096 elements, ~kFLOP, ~µs of compute).
The win is from de-serializing 86 norms/step (43 layers × 2 sublayers) off
GPU0 — replacing 86 GPU0-serial kernels with 86 × (8-way parallel kernels +
tiny all-reduce + all-gather of normed). Ballpark: **~5–15 ms/step shaved ⇒
~0.5–1.5 % tok/s**.

The bigger payoff is opening the door to:

- Same pattern applied to compressed KV, attention state, post-attn FFN-input
  norms (each a separate sprint, each ~0.5–1.5 %).
- Cumulative attention-staging de-centralization: **realistic 3–6 % tok/s**
  across the follow-on sprints.

Each one passes through the same gate, the same template, the same diagnosis
script — once #1's bug is fixed, the rest are mechanical clones.

## Reporting

For the A/B run report:

- Selected-token agreement (gating).
- Generated-sequence agreement (gating).
- Max selected-logit rel-err (advisory only — do not act on the value).
- Decode tok/s, projected slot-step tok/s (control vs candidate).
- Request-window GPU util (expect GPU0 share to drop, other ranks to rise).
- Per-fine-bucket ms (especially `pre_ep_attention_projection` and the related
  staging buckets) — the wins should appear there.
- `peer_copy_sys_bytes` (must remain 0).

## One-line summary

The existing `--attention-projection-rank-local-input` path is missing the
`ncclAllReduce` of the partial `Σx²` — every rank normalizes against its
local-shard sum instead of the global sum. Add the all-reduce, verify the
norm-weight sharding, all-gather the normed shards back for unchanged
consumers, gate on agreement-only, promote.
