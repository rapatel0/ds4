# Spike B Decode-Optimization Steering (2026-05-27)

Steering for the next TP/EP serving-throughput phase, off the de-confounded
steady-state reference (32 slots / 256K / 256 req / 64 tok/req, ~35.9 tok/s
server decode, ~889 ms decode domain).

## Unifying diagnosis — read this first

Both dominant domains are **overhead-bound, not compute- or bandwidth-bound:**

- **EP 53% (~473 ms):** at 32 slots × top-6 = 192 activations over 256 experts,
  each GPU's 32 local experts see **<1 token** → grouped GEMM is tile-underfill;
  the cost is dispatch / pack / reduce / all-to-all orchestration, not FLOPs.
- **HC-current 40% (~357 ms):** moves ~2 MiB → the time is launch-count + sync +
  **GPU0-centralization** × 43 layers, not data.

**Consequence:** wins come from STRUCTURE — de-centralize, fuse, fewer launches,
capture static sub-regions, more tokens-per-step (MTP) — **not** from faster
kernels. Optimizing the expert GEMM alone moves a fraction of 53% and none of 40%.

## A. HC-current (40%) — de-centralize GPU0 (highest-ROI, eager-path win today)

Steps 2–6 and 9–10 run on GPU0 over the full hidden while 7 GPUs idle. Every
"global over hidden" op decomposes into rank-local partial + one tiny all-reduce:

- **A1 RMS-norm rank-local:** partial sum-of-squares on each `[slots,4,512]`
  shard → all-reduce `[slots,4]` scalar → normalize locally. No gather.
- **A2 HC mix as row-parallel GEMM:** `attn_fn[16384×24] @ hc[slots,16384]` —
  16384=4×4096 is the contraction, already sharded (2048/rank). Each rank computes
  partial `[slots,24]` → all-reduce `[slots,24]` (3 KiB). Replaces gather-to-0 +
  full-norm + mix + split + broadcast with 2 rank-local kernels + 1 tiny allreduce.
- **A3 Router rank-local:** `[slots,4096]·[4096,256]`, 4096 contraction → row-
  parallel → all-reduce `[slots,256]`. (Sprint 426 started — finish, make default.)
- **A4 Drop the full-current allgather (step 8)** by making ALL consumers
  rank-major (attn-proj done = +13%; do router + FFN-norm + post-attn FFN input).
- **A5 Fuse the survivors:** after A1–A4, HC-current ≈ 3 rank-local kernels + 2
  tiny all-reduces/layer vs ~12 steps. Fuse norm+partial-mix; mix-apply+FFN-norm.
- **A6 Fuse HC into the attention-projection prologue** (compute `current_shard`
  inside the projection kernel; drop the intermediate buffer + a launch).

Net: GPU0-serial / ~12-launch / barrier-heavy → 8-way parallel / ~3-launch / 2
tiny collectives. **A1–A3 is the concrete "how" for the 40%.**

## B. EP (53%) — orchestration + sub-1-token experts

- **B1 MTP is an EP-efficiency lever, not just a decode multiplier.** EP is 53%
  *because* M<1/expert; MTP (verify K draft tokens) makes experts see (K+1)× tokens
  → tiles fill + fewer steps. Sequence after correctness is stable, but it is the
  structural fix for the largest domain, not optional polish.
- **B2 Fuse dispatch + grouped-GEMM + weighted-combine** into 1–2 kernels,
  device-side offsets only (no host sync on route counts). Template: the fork's
  `awq_moe_single_token_sm70` compact path. Make compact-route-compose the default.
- **B3 TP-sharded experts vs EP A/B (the S-F question — now justified).** EP's 53%
  is dispatch/all-to-all. TP-experts have **no all-to-all** (reduce via the hidden
  all-reduce). For 13B-active/8-GPU/32-slot, test whether all-to-all overhead >
  TP-reduce cost. Strongest evidence yet to actually run it.
- **B4 Overlap shared (dense, rank-local) expert with the routed all-to-all
  dispatch** (independent until combine; re-apply sprint-261 overlap on serving).
- **B5 Correctness-preserving capacity balancing** (capacity-16 broke output in
  435; need a parity-clean fixed-shape scheme — also helps capture, see C1).

## C. Cross-cutting — and the highest-ceiling idea

- **C1 Piecewise graph capture (likely the biggest single win).** Full-serving-
  graph is blocked on parity + dynamism, but the per-layer compute region
  (HC-current → attention → EP → compose) is proven capturable in the smoke. Graph
  ONLY that static sub-region; run dynamic orchestration (request mgmt, output
  head, sampling) eager around it; feed routing via **persistent device buffers**
  (fixed worst-case `slots×top_k`; route values change per step, graph structure
  fixed). Bridges "works in smoke, blocked in serving" → could recover most of the
  2.27× without solving the whole loop. The async-route-plan was the capture-
  breaker → make routing write a persistent device buffer the graph reads.
- **C2 Fix the graph-in-serving parity bug directly.** Graph mode changes the
  first token = a finite set of missing sync→event dependencies (461 fixed one).
  Diff eager vs graph dependency graph; close them all. Debuggable, not fundamental.
- **C3 Launch-count reduction is the universal eager-path lever.** Every fusion
  above (A5/A6/B2) helps the eager path that serves today, graphs or not. Given
  graph-in-serving is blocked, this is the only lever lifting the served ~12% util
  right now.
- **C4 Run the spill check** (still not done) on the HC-mix kernel, head_dim-512
  attention, and rank-major consume kernels: `-Xptxas -v` (registers/smem/spill) +
  ncu (occupancy, long-scoreboard stalls). If they spill, the rank-local rewrites
  leave perf on the floor.

## Recommended priority

| # | Idea | Domain | Why | Risk |
|---|---|---|---|---|
| 1 | A1–A3 rank-local HC norm/mix/router | HC 40% | De-centralizes GPU0; 8-way parallel; eager-path win now | Low |
| 2 | C1 piecewise graph of layer compute | both | Highest ceiling — recovers stranded 2.27× in serving | Med |
| 3 | B3 TP-experts vs EP A/B | EP 53% | Profile justifies it; could delete the all-to-all | Med |
| 4 | A5/A6 + B2 kernel fusion / launch-count | both | Universal eager win; compounds with #1 | Low |
| 5 | B1 MTP | EP 53% | Structural fix for sub-1-token experts; biggest multiplier | Med (after #1–4) |

## Discipline (unchanged)

- A/B every idea on the steady-state reference above, **parity-gated** (first
  token unchanged + tolerance) — not on reduced/short shapes that flatter results.
- Report **server decode tok/s AND GPU util** for control vs candidate.
- No re-baselining; no micro-opt before the change is justified by the profile.
- Re-profile the domain table after each promoted change (shares shift).

## One-line frame

Both big domains are "do less / in parallel / in one launch" problems — **#1
(rank-local HC) + #4 (fusion) lift the eager served path now; #2 (piecewise graph)
is the swing for the big win; #3 and #5 are the structural EP bets.**
