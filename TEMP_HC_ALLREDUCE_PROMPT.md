# Steering Prompt — HC-current rank-local all-reduce (kill GPU0 broadcast + SYS)

Implements Section A (A1/A2/A3/A6) of `SPIKE_B_STEERING.md`, now with the
empirical SYS evidence that makes it priority #1.

## Objective

Convert HC-current's three "global-over-hidden" reductions from
**gather-to-GPU0 → compute-on-GPU0 → peer-broadcast** into
**rank-local partial → `ncclAllReduce` of a tiny vector → finish locally**.

Three concrete, parity-preserving wins:
1. Eliminate the GPU0 serialization in the ~40% HC-current domain (8-way
   parallel instead of GPU0-serial while 7 ranks idle).
2. Eliminate the SYS-crossing `ds4_peer_copy_async` GPU0→rank broadcasts
   (the instrumentation below shows 256 MB of Direct SYS traffic from them).
3. Drive toward **never materializing the full `[slots,4096]` current hidden**:
   convert consumers to rank-local so the full-current `ncclAllGather` @7495 is
   deleted (A4 — staged per consumer; this is the bigger surface).

This lifts the **eager** served path (no graph dependency) and, because every
new cross-rank op is an NCCL collective (graph-capturable), it also unblocks
C1 piecewise graph capture later. No EP/MoE, MTP, or graph-capture work here.

## Evidence (s478 peer-accounting run, 32 slots / 256K / r4)

| Metric | Value |
|---|---:|
| NCCL graph SYS edges | **0** (collectives use the NVLink ring) |
| Direct peer-copy ops / bytes | 82,389 / 593 MB |
| **Direct SYS ops / bytes** | **35,313 / 256 MB** |
| Top SYS site `run_true_ds4_attention_projection_prefix:12817` | 67.6 MB SYS, 129 ops |
| First SYS site `run_shared_hc_current_input:7454` | 3072 B/copy |

**Root cause:** V100-SXM2 NVLink is not full all-to-all. `ds4_peer_copy_async`
from GPU0 to a non-NVLink-adjacent rank falls back to PCIe/SYS. NCCL ring/tree
respects the topology → 0 SYS. **Therefore: cross-rank data moves only via NCCL
collectives, never a hand-rolled GPU0 peer broadcast.**

## The math identity (why this is parity-safe)

Each rank holds its `[slots, 4, 512]` shard = 2048 of the 16384 flattened-HC
contraction columns. The two HC-pre reductions are **additive over the feature
axis**, so a global reduction == sum of rank-local partials:

- **RMS scale:** `Σ_{16384} x²  =  Σ_ranks (Σ_{local 2048} x²)`.
- **Mix GEMM:** `mix[24] = Σ_{16384} attn_fn[:,c]·x[c]
                       = Σ_ranks (Σ_{local 2048} attn_fn[:,c]·x[c])`.
  And the RMS scale is a per-slot scalar that **factors out** of the GEMM:
  `mix = scale · (attn_fn @ x_unnormed)`. So you can reduce the *unnormalized*
  partial GEMM and the partial sum-of-squares, then combine locally.

### A2 — mix + RMS as one rank-local + one tiny all-reduce (do this FIRST; deletes site 7454)

Replace the block at ~7420–7465 (and the twin at ~7090) for the gated path:

1. **Reshard `attn_fn`/`ffn_fn` by contraction column at load.** `attn_fn` is
   `[24, 16384]` fp32 (1.5 MB/layer). Each rank gets the 2048 columns matching
   its hidden shard. ⚠ **Layout trap:** the 16384 is `[row0:4096][row1][row2][row3]`,
   so a rank owns **four strided 512-blocks** (cols `row*4096 + rank*512 + 0..511`
   for row=0..3), not one contiguous 2048-block. Build the slice accordingly.
2. Per rank, compute **partial sum-of-squares** `[slots]` over its 2048 shard and
   **partial unnormalized mix** `attn_fn_shard[24,2048] @ x_shard → [slots,24]`.
3. **One grouped `ncclAllReduce(ncclSum, r.compose_nccl)`** of `[slots, 1+24]`
   (sumsq ‖ partial-mix) — 3.2 KB at 32 slots, on the NVLink ring (0 SYS).
   Wrap the per-rank issue in `ncclGroupStart/End`.
4. Each rank locally: `scale = rsqrt(sumsq/16384 + eps)`, `mixes = scale·mix`,
   then run `hc_split_rows_kernel` on its own `mixes` → `d_hc_split` resident.
   **No gather, no GPU0 norm/mix/split, no broadcast (7454 gone).**
5. `hc_weighted_sum_shard_kernel` unchanged (already rank-local).

⚠ **Parity trap (stable RMS):** `rms_norm_plain_rows_stable_kernel` rescales by
global `max_abs` for numerical stability. To match it exactly, also all-reduce a
`[slots]` global max-abs (`ncclMax`) and reproduce the stable formula; or accept
the plain formula within the parity tolerance. Decide and document which.

### A6 — attention-projection rank-local default (deletes site 12817)

The rank-local branches already exist (`rank_major_input` @12776 →
`d_current_full_rank_major`; `direct_input_fill` @12801; `d_current_full_normed`
@12790). The 67 MB SYS site is only the `else` fallback (12807–12821) that
broadcasts `hc->d_attn_normed` from GPU0. Promote the rank-local/rank-major path
to **default** once parity-clean so the GPU0 broadcast is never issued. Produce
the FFN/attn norm rank-local too (partial sumsq → all-reduce `[slots]`), so
`d_attn_normed`/`d_current_full_normed` exists per-rank without GPU0.

### A3 — router all-reduce (after A2 lands)

Router logits = `current @ W_router` contracting over 4096 → rank-local partial
`[slots,256]` → `ncclAllReduce([slots,256] = 32 KB)`; top-k select runs on full
logits locally. There is already `ncclAllGather(d_router_logits_shard)` @7155 —
evaluate whether all-reduce of the partial logits is cheaper than
allgather+recompute. Apply the `noaux_tc` bias / hash-router path **post-reduce**,
identically to today, and parity-check it.

### A4 — Remove the full-current allgather; consumers go rank-local (the bigger surface)

A2/A3 remove gather-*for-reduction*. Separately, step 8 builds the full
`[slots,4096]` current hidden on every rank via
`ncclAllGather(d_current_shard → d_current_full_rank_major)` @7495 (+ slot-major
transpose @7530), with a `ncclBroadcast(d_current_full)` @12750 and GPU0 peer
copies @7727/7767/9811/12817. This exists only because the **consumers** still
expect a full hidden vector:

| Consumer | Site | Input |
|---|---|---|
| Attention Q/KV projection | 12790/12822 | full normed current (A6 converts this) |
| Post-attention dense + shared FFN input | 7735/7741/7747/7754 | `r.d_current_full` |
| Shared-expert gate/up FFN | 9818/9825 | `r.d_current_full` (ffn_normed) |
| Routed-expert input packing | 7776/7781 | `r.d_current_full` |

Two levels, staged by risk:

- **A4a (low risk, do with A6):** for every consumer still pulling full current
  via `ds4_peer_copy_async` from GPU0 (7727/7767/9811/12817), route the cross-rank
  build through the existing **NCCL** rank-major path (`d_current_full_rank_major`)
  instead. Deletes the remaining Direct-SYS peer copies even where the consumer
  still wants a full vector. (rank-major = full hidden still materialized, but
  built by a topology-clean NCCL allgather, no GPU0 SYS, no slot-major transpose.)
- **A4b (bigger win, per-consumer):** convert each consumer GEMM to **row-parallel**
  (input sharded 512/rank) so it consumes only its local shard and **all-reduces
  its output**, deleting the full-current allgather @7495 for that consumer.
  Per-consumer tradeoff: row-parallel pays an output all-reduce instead of an
  input gather — measure which is smaller for each GEMM
  (q_lora 1024 / kv_lora 512+rope / moe_inter 2048 / expert-pack). Requires
  resharding that consumer's input weight by row + a per-consumer parity gate.
  Precedent: attn-proj rank-local (+13%); router+FFN rank-major (sprint 451,
  ~1.03–1.04×).

Delete the allgather/broadcast (7495/12750) only once **all** consumers are
converted; until then keep it but force it through NCCL (A4a), never GPU0 SYS.

### A5 — Fuse the survivors

After A1–A4, HC-current per sublayer ≈ 2 rank-local kernels + 2 tiny all-reduces.
Fuse norm+partial-mix into one kernel, and mix-apply+FFN-norm into one, dropping
the per-layer launch count from ~12 to ~3. This is the eager-path launch-count
lever that serves today regardless of graph capture.

## Deliverables & gating

1. Default-off gate, e.g. `--hc-current-allreduce` /
   `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`, wrapping A1+A2. Reuse/extend the
   existing attention-projection-rank-local gate for the 12817 path.
2. Reuse `r.compose_nccl`; `ncclGroupStart/End`; **warm up each new all-reduce
   before any graph capture** (collectives are capturable; peer copies are not).
3. Reshard `attn_fn`/`ffn_fn`/router weight by contraction column at load.
4. **Parity:** selected-token parity 4/4 vs control on the reference shape, plus
   a direct fp32 diff of all-reduced `mixes` vs GPU0 `d_mix` within tolerance
   (reduction order differs) — same discipline as sprint 427's half-input audit.
5. **A/B on the steady-state reference only** (32 slots / 256K / 256 req /
   64 tok). Report, control vs candidate: server decode tok/s, request-window
   GPU util, **and Direct SYS bytes/ops** (target: 7454 + 12817 SYS → 0).
   No reduced/short shapes.
6. Re-profile the HC-current domain table after each promotion (the 84 ms
   `pre_ep_hc_current`, 41 ms `ffn_router`, 28 ms `route_upload` fine buckets
   should shrink; shares shift toward EP).

## Order

1. **A2** mix+RMS all-reduce → deletes 7454 → measure.
2. **A6 + A4a** attn-projection rank-local default, and route all remaining
   full-current GPU0 peer copies through NCCL → deletes 12817 + the 7727/7767/9811
   Direct-SYS copies → measure (Direct SYS should approach 0 here).
3. **A3** router all-reduce → measure.
4. **A4b** convert consumers to row-parallel one at a time (post-attn FFN, shared
   FFN, expert pack), each parity-gated; delete the full-current allgather @7495
   only after the last consumer is converted → measure.
5. **A5** fuse the survivors → measure.

Each step parity-gated and SYS-accounted independently; promote only what
survives both parity and a measured win on the reference shape.

## Out of scope

EP/expert *compute* path, MTP, and graph capture itself (C1) — though deleting
the allgather/peer-copies here is what later unblocks C1. Constraint: keep every
new cross-rank op an NCCL collective so it stays capturable and SYS-clean.
