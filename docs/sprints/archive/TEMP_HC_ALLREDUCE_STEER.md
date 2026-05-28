# Steering — HC-current rank-local all-reduce (in-flight)

You are already executing `TEMP_HC_ALLREDUCE_PROMPT.md`. This is a steering
overlay: keep to this priority order and these gates. Don't re-plan.

Reference shape for all A/B and parity: **32 slots / 256K / 256 req / 64 tok.**
Promote a step only if it passes parity AND shows a measured win there.

## Priority order

1. **A2 — mix + RMS all-reduce (do first; highest ROI, lowest risk).**
   Per rank: partial sum-of-squares `[slots]` + partial *unnormalized* mix
   `[slots,24]`; one grouped `ncclAllReduce([slots,25], ncclSum, r.compose_nccl)`;
   then `scale = rsqrt(sumsq/16384+eps)`, `mixes = scale·mix`, split locally.
   Deletes the GPU0 gather/norm/mix/split and the split broadcast at **7454**.
   Land, measure, then proceed. (Watch the two traps: the `attn_fn` reshard owns
   four strided 512-blocks, not one contiguous 2048; and match the *stable* RMS
   via a `[slots]` `ncclMax` of max-abs, or accept the plain formula within tol.)

2. **A6 + A4a — kill the targeted full-current Direct-SYS peer copies
   (mandatory, not optional).**
   Make attention-projection rank-local the default (deletes **12817**), and route
   every remaining full-current GPU0 peer copy (**7727 / 7767 / 9811**) through the
   NCCL rank-major path. After this, the full-current Direct-SYS sites should be
   near zero. Total Direct-SYS may remain nonzero from other surfaces until those
   are separately converted.

3. **A3 — router all-reduce.** Partial logits `[slots,256]` → `ncclAllReduce` →
   top-k locally; apply `noaux_tc` bias / hash-router path **post-reduce**.

## GUARDRAIL — A4b is a per-consumer bet, not a mandate

A4b (convert consumers to row-parallel, delete the @7495 allgather) trades a
**shared** input gather for **per-consumer** output all-reduces. At 32 slots
(latency-bound), more-but-smaller collectives can *regress*. Rules:

- Convert and measure **one consumer at a time**; keep only measured winners.
- Expect narrow-output consumers to win (kv_lora 512+rope), wide-output to wash
  or lose (FFN gate/up @ moe_inter 2048). Precedent: attn-proj rank-local **+13%**;
  router+FFN rank-major only **~1.03–1.04×** (sprint 451).
- Do **not** delete the @7495 allgather / 12750 broadcast until *every* remaining
  consumer is converted — a half-conversion pays both and regresses.
- If A4b nets **< ~3%** after the obvious narrow wins, **stop**: leave the shared
  gather (now NCCL/SYS-clean from A4a) in place. A2 / A6 / A4a / A3 are bankable;
  A4b is opportunistic.

## Gates (every step)

- **Parity:** governed by `TEMP_PARITY_POLICY.md`.
  - Arithmetic-changing steps are any steps that reorder or recompute a
    reduction, including A2/A3/A4b and A6 rank-local attention norm/input:
    **tolerance gate, not bit-exact** — fp64-accumulate the reduction where
    applicable, then teacher-forced top-1 ≥ 99% (or logit rel-err ≤ 1e-3) +
    coherence. Pass tolerance but fail exact match → **promote**.
  - Transport-only steps keep the same arithmetic on the same device and only
    replace direct peer movement with NCCL. These stay bit-exact selected-token
    vs control; a mismatch there is a real bug, not drift.
- **Report** control vs candidate: server decode tok/s, request-window GPU util,
  **Direct SYS bytes/ops**. Reference shape only — no reduced shapes.
- Every new cross-rank op is an **NCCL collective** (capturable, SYS-clean),
  warmed up before any capture. Never a GPU0 `ds4_peer_copy_async`.
- Prefer NCCL for every cross-rank reduction. Use topology-clean defaults
  (`NCCL_P2P_LEVEL=NVL` and the no-SYS ring hint), but leave
  `NCCL_ALGO`/`NCCL_PROTO` on `auto` unless a measured candidate benefits from
  pinning. For tiny HC/router reductions, explicitly test `Tree+LL128`; for
  larger bandwidth-bound collectives, let NCCL tune or measure Ring.
- Re-profile the HC-current domain table after each promotion.

## Stop condition

Stop when the targeted full-current **Direct SYS ≈ 0** and the HC-current domain
share has dropped on the reference profile. Report the new domain table
(EP / HC-current / rest) plus residual Direct-SYS sites so we re-pick the next
target.

## Current Promotion Notes

- A2 is promoted in appliance defaults.
- A4a targeted full-current transport cleanup is promoted at code level after
  V100 build and short 32-slot / 256K peer-accounting smokes.
- Residual Direct-SYS is expected and currently points at router-plan upload and
  shared/EP materialization, not the full-current broadcasts targeted here.
