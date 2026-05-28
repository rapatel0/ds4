# Steering — post-SYS-sweep optimization docket

The work queue for after sprint **479** (SYS Transport Sweep) lands. Ordered by
impact-effort quadrant. Excludes:

- **Sprint 479 itself** (in flight) — every transport peer-copy → NCCL replacement.
- **A2 mix/RMS all-reduce** (sprint 478) — already implemented; will re-judge
  under the tolerance gate (`TEMP_PARITY_POLICY.md`) after 479. Treat as
  "queued for promotion," not new work.
- **A6 `--attention-projection-rank-local-input`** — already implemented but
  failed bit-exact gate because it recomputes the attention norm per rank;
  it's actually arithmetic-changing, queued under tolerance like A2.

## Calibration (to ballpark magnitudes)

Reference shape: 32 slots / 256K / 256 req / 64 tok. Baseline server decode
**35.8 tok/s, ~12 % util, HC-current 358 ms, EP 473 ms** of 893 ms/step.

Measured anchors:
- **A2 (one small all-reduce per sublayer × 43 layers):** +2.7 % tok/s,
  HC-attn-mix 20.83 → 10.03 ms.
- **Attn-projection rank-local (one Pattern B narrow consumer):** +13 % tok/s.

Heuristic: each Pattern A sub-step is ~**+1–3 %**; each Pattern B narrow
consumer is ~**+5–13 %**; structural items (graph / MTP / TP-experts) are
**multiplicative**.

## The FP8/MXFP4 packing reality (shifts Pattern B effort up)

Weights are **MXFP4** (experts, 32-elem groups along contraction) and **FP8**
(dense, typically 128-tile). Dequant is **fused into the SM70 WMMA GEMM kernel**
(TurboMind). For any **row-parallel** (A4b) candidate, audit before coding —
this is where the real cost lives, not in the call-site rewrite:

1. **Group alignment.** Shard = 4096/8 = **512**. Need 512 ≡ 0 (mod group).
   MXFP4 (32) → 16 groups/rank ✓. FP8-128-tile → 4 tiles/rank ✓. Anything not
   a multiple of 32/64: FAIL — repack required.
2. **Activation scale topology.** Per-group / per-token: safe, rank-local.
   **Per-tensor or per-channel-row scaling: BREAKS row-parallel** (scale
   depends on the unsharded vector).
3. **Dequant-GEMM kernel K-shape.** The fused kernel is tuned for K=4096
   contraction. Row-parallel feeds it **K=512** — 8× less pipelining depth
   for the dequant. May regress per-rank GFLOPS even though total FLOPs ÷ 8.
   **This is the dominant effort cost** and is the per-consumer kernel-tune
   you actually buy.
4. **Output requantization.** Inter-layer transport is fp16/bf16, not packed,
   so the all-reduced output is consumed JIT-quantized by the next op exactly
   like today. No extra cost beyond the all-reduce itself.

Implication: my prior "+13 % attn-proj" precedent demonstrates Pattern B is
*feasible*, but extending it to each new consumer is roughly **one kernel-tune
sprint per distinct dequant-GEMM shape**, not a call-site rewrite.

---

## Quadrant: HIGH IMPACT / LOW EFFORT — do first

| # | Change | Pattern | Magnitude | Effort | Notes |
|---|---|---|---:|---|---|
| 1 | **A2 re-promotion** (mix/RMS rank-local + all-reduce) | A | **+2.7 % (measured)** | trivial — flip the gate, re-run under tolerance | implemented in 478; passes per-layer ~1e-6; only failed bit-exact gate. fp64 partial accumulation as risk-reducer (`TEMP_PARITY_POLICY.md`). |
| 2 | **A3 router all-reduce** (`[slots,256]`) | A | **+2–3 %** | low — same template as A2 | router fine-bucket = 41.7 ms (4.7 %); rank-local partial → all-reduce → top-k locally. Apply `noaux_tc`/hash bias *post-reduce*. |
| 3 | **A1-attn + A1-ffn rank-local norms** (RMS over 4096) | A | **+0.5–1 % combined** | low — same template, smaller all-reduce (`[slots]`) | the bug that bit A6 — promote *intentionally* under tolerance gate, not snuck in as "transport." |
| 4 | **EP-compose `ncclReduceScatter` promotion** (mode b → default) | C | **+1–3 %** | trivial — flip existing `nccl_reduce_scatter_compose` gate | fuses alltoall + kernel-sum into one NCCL. Tolerance-gated because NCCL tree-order ≠ kernel `for src=0..7`. |
| 5 | **Final RMS-norm before LM head** rank-local | A | **+0.1–0.3 %** | trivial — one-shot per step | A1 pattern at the model boundary; easy to forget because it's outside the layer loop. |

**Quadrant subtotal: ~+6–10 % tok/s.** Each is the same A2 template (one
small `ncclAllReduce` + local finish) or a gate flip, runs under the tolerance
gate, low risk, completable in one short sprint each.

## Quadrant: LOW IMPACT / LOW EFFORT — opportunistic fillers

| # | Change | Pattern | Magnitude | Effort | Notes |
|---|---|---|---:|---|---|
| 6 | **LM-head argmax** via local argmax + `ncclAllReduce(MAX_LOC)` | D | **+0.5–1 %** | low — one all-reduce per step | replaces a vocab-wide (~516 KB) gather with a tiny reduce. Bit-exact-modulo-tie-order; under "tolerance" only by convention. |
| 7 | **Indexer top-K merge** (if currently GPU0-centralized) | D | **+1–2 %** if applicable | low — verify code first | full-K-per-rank merge is exact; check that the existing path isn't already rank-local before scheduling. |

Take these when convenient; not worth a dedicated sprint.

## Quadrant: HIGH IMPACT / HIGH EFFORT — the big bets

| # | Change | Pattern | Magnitude | Effort | Notes |
|---|---|---|---:|---|---|
| 8 | **A4b narrow attention consumers** row-parallel: `q_a` (1024), `kv_latent` (576), `o_a` (1024) | B | **+10–20 % combined** | **high per consumer — FP8 dequant-GEMM K=512 kernel re-tune** | precedent: +13 % on one consumer. Each is a separate parity gate. Skip wide-output siblings (`o_b`/FFN/`down`) — predicted regression. |
| 9 | **Piecewise graph capture (C1)** | — | **+50–100 % potential** (recovers stranded 2.27× graph win) | **very high** — parity in graph mode, persistent device buffers, dynamic routing fed via persistent buffer | unblocked by 479: all hot ops will be NCCL-capturable. The async-route-plan was the prior capture-breaker. |
| 10 | **MTP — multi-token prediction (B1)** | — | **+30–100 % effective tok/s** | **very high** — speculative draft/verify, accept/reject across ranks, parity-as-distribution | structural fix for EP's M<1-token-per-expert (each step's experts see K+1× tokens). Also raises tokens-per-step linearly. |
| 11 | **TP-experts vs EP A/B (B3)** | — | **±10–30 % uncertain** | **very high** — major redesign; experts sharded by TP (no all-to-all), reduce via the hidden all-reduce | only justified if EP's share is still dominant after #1–4 + #8 + #10. If MTP fixes M<1, this may not pay off. |

## Quadrant: LOW IMPACT / HIGH EFFORT — defer or skip

| Change | Why skip |
|---|---|
| A4b wide-FFN row-parallel (shared/routed FFN gate / up / down at `moe_inter` 2048) | output all-reduce ≈ 2× input gather → byte math predicts regression. The FP8 dequant kernel re-tune doesn't recover that. **Hard skip** unless byte math changes (e.g., much larger batch). |
| LM head row-parallel | vocab 129,280 — all-reduce dominates everything else combined. **Hard skip.** |
| A4b shared expert (gate/up/down) | wide-output ≈ wash even before the FP8 kernel cost. Not worth it. |
| Sinkhorn rank-distribute | 24 numbers per token, redundant per rank is already free. No-op. |

---

## Suggested phasing

- **Phase 1 (immediately after 479):** items **1–5** (Pattern A re-promotions
  + gate flips). One sprint, one tolerance A/B per item, expected
  **~+6–10 %** combined. Tolerance gate becomes "the way we work" here.
- **Phase 2 (optional):** items **6–7** if cheap; skip if not.
- **Phase 3 (the per-consumer Pattern B work):** item **8**, one consumer per
  sprint, each its own tolerance + perf A/B. Pre-each: audit the four FP8/MXFP4
  checks above and budget for the K=512 kernel re-tune. Expected **+10–20 %**
  cumulative.
- **Phase 4 (re-pick from the profile):** re-profile after Phase 1+3. If EP
  share has shrunk, defer #11; if it's still dominant, **#10 (MTP)** is the
  structural play before #11.
- **Phase 5:** **#9 piecewise graph capture.** Only attempt after 479 + Phase 1
  have eliminated peer-copy from the hot path; the capture-breakers were
  exactly those non-capturable peer copies plus the async-route-plan.

## Out of scope (for this docket)

- Sprint 479 (in flight).
- A2 / A6-norm — already implemented, queued for tolerance-gate promotion as
  items #1 / #3 above.
- Anything that requires changing the parity gate from what
  `TEMP_PARITY_POLICY.md` already defines (it covers both tolerance and
  bit-exact use cases).
- Spike A (vLLM port) — paused, separate program.

## One-line summary

After 479: **flip A2's gate, then bank ~+6–10 % from Pattern A/C tolerance
sprints, then ~+10–20 % from Pattern B narrow-attention with kernel re-tune,
then commit to ONE big bet — graph (C1) or MTP (B1) — based on the re-profile.**
