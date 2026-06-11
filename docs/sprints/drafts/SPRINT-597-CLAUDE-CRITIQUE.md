# Sprint 597 Draft Critique (Claude)

Critique of `SPRINT-597-CODEX-DRAFT.md` and `SPRINT-597-GEMINI-DRAFT.md` against
`SPRINT-597-INTENT.md`. Every code-level claim either draft makes was checked
against the actual source on this branch before being accepted or rejected; the
verification ledger is at the end.

## TL;DR

- **Codex draft**: substantially stronger. It is the only one of the two that
  actually opened the source files, and as a result it corrects a real error in
  the intent's line map, gets the TurboMind ABI story exactly right, and ships a
  staged implementation plan with gate flags, files, and per-stage Definition of
  Done. Its main weakness is the opposite of Gemini's: it is long and front-loads
  Phases 4–7 (the B2 stages) that the intent says belong in 598+, blurring the
  "instrumentation + decision only" boundary it otherwise argues for.
- **Gemini draft**: correct in its high-level shape and conclusions, but thin and
  partly unverified. It inherits the intent's stale line map without checking it,
  designates `nsys` graph capture as the "authoritative gate" without confronting
  the fact that the existing in-band timers do not populate in graph mode (the
  central instrumentation problem), and its one stated risk misnames the eager
  sync primitive. It reads like a faithful summary of the intent rather than an
  independent verification of it.
- Both drafts reach the same correct ABI conclusion (need a new device-resident
  entry point or a full-shape masked executor) and the same packaging answer
  (597 = instrument + decide, 598+ = stages). The differentiator is depth and
  whether the claims were verified.

---

## Verification of code-level claims (summary; full ledger at end)

The contested technical claims resolve as follows (file:line evidence below):

1. **Codex's "the intent line map is slightly stale" — CONFIRMED.** The intent
   says routed FFN is followed by `sync_all()` at line ~918 and another at
   ~1238. In the actual source, `decode_loop.cu:918` is
   `sync_after_decode_stage("routed_ffn")` and `decode_loop.cu:1238` is
   `sync_after_decode_stage("ep_copy")` — **not** `sync_all()`. These are an
   *opt-in* stage sync (`decode_loop.cu:210-237` early-returns unless
   `decode_cudagraph_gate` is on **and** the stage is named in
   `decode_cudagraph_stage_sync`). The unconditional `sync_all()` barriers are at
   lines 954, 996, 1062, 1144, 1170, 1373. Codex caught this; Gemini repeated the
   intent's framing ("Global `sync_all()` barrier waits") without correction.

2. **Both drafts' TurboMind D2H-read claim — CONFIRMED.** The base
   `ggml_turbomind_mul_mat_grouped()` does `cudaMemcpy(&total_tokens,
   &expert_offsets[num_experts], …, cudaMemcpyDeviceToHost)` at `api.cc:1078`.
   The `_total_tokens` variant (`api.cc:1173+`) takes a host `int total_tokens`
   and skips the read. Both drafts state this correctly.

3. **Codex's "DS4 binds only `mmgt`/`mmgs`/`mmgs_clamped`" — CONFIRMED and
   materially sharper than Gemini.** `turbomind_bindings.cu:52-55` loads only the
   three `_total_tokens` symbols. **This means DS4 already avoids the base API's
   internal D2H read** — the host round-trip that actually remains is in
   `router_plan.cu:70-75`, where `d_route_totals`/`d_route_offsets_all` are copied
   D2H to set `rank.routes`, which `routed_executor_rows()`
   (`turbomind_bindings.cu:374`) returns as the GEMM's `executor_rows`. Gemini's
   Q4 answer attributes the D2H cost to "the base API" without noting DS4 doesn't
   call the base — its conclusion is right but its diagnosis points at the wrong
   line.

4. **CUDA-graph barrier mechanics — CONFIRMED.** `sync_all` (`decode_loop.cu:174`)
   is a cross-rank event barrier (`enqueue_cross_gpu_stream_barrier`) under
   `decode_cudagraph_gate`, else a per-rank `cudaStreamSynchronize` over
   `stream` + `dense_stream`. The intent and Codex describe this accurately.

5. **Gemini's eager-timer risk claim — PARTLY WRONG.** Gemini says eager timers
   "rely on `cudaDeviceSynchronize()`." The eager path uses per-stream
   `cudaStreamSynchronize` (`decode_loop.cu:186`), not `cudaDeviceSynchronize`.
   Minor, but it is the only technical assertion in Gemini's Risks section and it
   is imprecise.

6. **Codex's compose/broadcast/no-SYS references — ALL CONFIRMED.**
   `broadcast_ep_return_slices` (`runtime_pack.cu:267`) loops sources, uses
   `ncclBroadcast` + per-destination `cudaMemcpyAsync` into
   `d_ep_remote[src]`/`d_ep_remote_half[src]` (the fp16 path exists).
   `PeerCopyAccounting` (`runtime_types.cuh:92`) and `v100_nvlink_count`
   (`runtime_types.cuh:140`) exist as claimed. `compose_next_hidden_compact8_multi_kernel`
   (referenced in Codex Phase 5) exists in `kernels/v100/compose.cuh`.
   `upload_post_attention_fixed_capacity_route_plan_gpu` sets
   `routes=route_capacity`, `active_experts=kLocalExperts` and skips the totals
   readback (`router_plan.cu:198-200`) — Codex's "correctness-clean but slow"
   characterization is accurate.

Net: Codex made ~8 verifiable source claims and all 8 hold. Gemini made ~3, two
hold and one (the eager primitive) is imprecise; more importantly Gemini did not
independently verify the intent's line map and so carried its error forward.

---

## Codex draft

### Strengths

- **Verified against source.** It is the only draft that demonstrably opened the
  files. The line-map correction (point 1 above), the ABI binding detail (point
  3), and the `api.cc` D2H/`_total_tokens` split are all correct and not derivable
  from the intent alone.
- **Sharp ABI conclusion with the decision made explicit** (Q4 / "B2 Target
  Shape"): either a new ABI taking device totals/masks, or a DS4 full-shape
  executor that keeps `total_tokens = route_capacity` and early-exits inactive
  rows. This is the correct framing and it names the real cost of the cheap
  option (the fixed-capacity path already exists and is "correctness-clean but
  slow because it launches the full envelope").
- **Refuses to oversell host-readback removal.** Q3 and Phase 3 both say: if the
  measured host-sync bucket matches Sprint 581's ~5%, do not market readback
  removal as the perf win — treat it as a graph-capture *prerequisite*. This is
  the single most important piece of intellectual honesty in either draft and it
  directly answers intent Open Question 3.
- **Decision gate is genuinely data-driven** (Phase 3): explicit branch logic
  keyed to which stage dominates (GEMM < ⅓ → don't fuse first; transfer dominates
  → sparse fp16 + one-hop; barrier dominates → pairwise events). This is what the
  intent's Success Criterion 2 asks for.
- **DoD is itemized and checkable** — names the exact artifacts (tolerance gate,
  control artifact, decode tok/s at reference shape, EP table, peer-SYS counters,
  selected-token + generated-sequence agreement, next-stage decision).

### Weaknesses

- **Scope bleed past the sprint boundary it argues for.** Codex correctly says
  597 should be instrument + decide, then includes fully-specified Phases 4–7
  (B2-A through B2-D) with file lists and per-phase tasks. The intent's
  packaging answer (and Codex's own Q1) put those in 598+. Either label Phases
  4–7 explicitly as "design contract for 598+, not 597 work" or move them to a
  separate cycle-design appendix. As written, a reader could execute Phase 4 in
  597 and violate the stated cut line.
- **Length / signal density.** At ~520 lines it restates the same ABI conclusion
  in Overview, Architecture, Q4, and Risks. The decision-relevant content would
  fit in half the space.
- **The eager-vs-graph gate question is asserted, not proven.** Codex picks the
  eager decomposition as the numerical gate "because the existing code only
  populates detailed per-stage timers in pure eager." That is the right call (see
  Gemini critique below), but Codex does not specify *how* the graph correlation
  check produces comparable per-substage numbers — i.e., whether Phase 1 adds
  `cudaEventRecord` nodes *into* the captured graph (which changes the graph) or
  relies on `nsys` kernel-timeline mapping. This is the actual hard part of the
  instrumentation and it is left implicit.

### Gaps in risk analysis

- **No quantified instrumentation-overhead budget.** The "instrumentation
  perturbation" risk is named but there is no acceptance threshold — e.g. "event
  recording must not move steady-state decode tok/s by more than X%, validated by
  a flag-on/flag-off A/B." Without it, the reconciliation-to-9.4ms gate
  (`DoD`) can't distinguish real attribution from measurement skew.
- **Self-consistency / closure risk is under-specified.** The intent demands
  sub-stages sum to ≈ the EP total. Codex's reconciliation phase allows the
  discrepancy to be "explained by changed graph/eager mode," which is a release
  valve that could absorb an arbitrary unattributed remainder. No bound on
  acceptable residual is given.
- **No risk around the eager baseline having drifted.** Sprint 581's 9.419 ms is
  the anchor, but the repo has moved since (MTP built and punted). Whether the
  current eager build still reproduces 9.419 ms is itself a risk — if it doesn't,
  the reconciliation gate has no fixed target.

### Missing edge cases

- The intent's Verification Strategy lists five edge cases (zero-token experts,
  all-tokens-to-one-rank skew, sub-capacity slot counts, graph replay with
  changing route counts, fp16 return precision vs the 0.99 gate). Codex's
  instrumentation phase does **not** carry these into the measurement plan — e.g.
  it should record the per-layer route-skew distribution so the decomposition is
  representative of worst-case routing, not just steady state. The edge cases
  reappear implicitly in the B2 stages but are absent from the part of the sprint
  that actually lands in 597.
- No mention of **how barrier-wait is attributed per rank** when the barrier is
  an event barrier in graph mode — "wait time" on an event barrier is
  GPU-side idle, not host wait, and the two require different measurement (event
  elapsed between arrival and release vs `cudaStreamSynchronize` host time). Codex
  gestures at "both per-rank host wait and stream-event elapsed time" for the
  eager case only.

### DoD completeness

Strong — the most complete of the three documents. The one gap: it does not
require recording an **instrumentation-on vs instrumentation-off throughput
delta**, which is the evidence that the attribution is trustworthy rather than
self-perturbed. Add that as a DoD line.

---

## Gemini draft

### Strengths

- **Correct shape and correct conclusions.** Every high-level answer (packaging =
  597 + 598+, both eager and graph profiling, ABI needs a new entry point,
  static one-hop schedule over mixed NCCL, supersede the README note) matches the
  intent and matches what the source supports. Nothing in it is *wrong* at the
  decision level.
- **Concise and readable.** The Open Questions section is a clean, scannable set
  of decisions. For an executive read this is the better document.
- **Names the eager-perturbation concern up front** as the headline risk and
  correctly ties the mitigation to using the captured-graph profile rather than
  trusting eager absolute latencies.

### Weaknesses

- **Largely unverified.** It repeats the intent's stale line map and primitive
  names without opening the files. It never catches that `decode_loop.cu:918`/`1238`
  are `sync_after_decode_stage` (opt-in), not `sync_all`. It cites
  `ep_pack_route_dest_shards_kernel` and `broadcast_ep_return_slices` straight
  from the intent. This is a summary, not an independent draft — which is a
  problem when the brief explicitly asks each model to "answer from the header,
  not assume."
- **The "authoritative gate" choice is backwards for the actual tooling, or at
  least unjustified.** Gemini declares the `nsys` full-capture profile the
  authoritative numerical gate. But the intent and the source both say the
  existing in-band per-stage timers **only populate in pure eager mode**; the
  promoted path is one captured graph whose internal substages `nsys` gives you
  as a raw kernel timeline you must hand-map across 8 ranks × 43 layers. Gemini's
  own DoD demands "a per-rank decomposition … comparing both eager and
  full-capture execution" — but it never explains how the per-substage
  full-capture numbers are produced (add event nodes into the graph? parse nsqm
  kernel names?). It picks the heavier tool as the gate without confronting that
  the heavier tool doesn't natively emit the table the DoD requires. Codex's
  inverse choice (eager = numerical gate, graph = correlation check) is better
  grounded.
- **ABI diagnosis points at the wrong line.** As noted in verification point 3,
  Gemini attributes the live D2H cost to the base `ggml_turbomind_mul_mat_grouped`,
  but DS4 binds only the `_total_tokens` variants — so the base's internal read
  isn't even on DS4's path. The host round-trip Gemini is implicitly worried
  about lives in `router_plan.cu`, not the GEMM. The conclusion (need
  `int* d_total_tokens` or persistent kernel) is right; the reasoning would
  mislead an implementer hunting for the readback.
- **The one technical risk is misstated** (`cudaDeviceSynchronize` vs
  `cudaStreamSynchronize`, verification point 5).

### Gaps in risk analysis

- **Only one risk is listed.** Missing entirely: graph-capture invalidation from
  inserting instrumentation, VRAM headroom (~30.7/32 GiB per the intent), no-SYS
  topology regression, eager-baseline drift since Sprint 581, and over-scoping.
  Codex's draft enumerates six; the intent flags most of these. For a sprint
  whose *named failure mode* is "structurally landed but perf didn't transfer,"
  a single-risk section is a significant gap.
- **No mitigation for the gate it chose.** If `nsys`-on-graph is authoritative,
  the risk that it cannot cleanly attribute barrier-wait/idle gaps (only kernel
  durations) is unaddressed.

### Missing edge cases

- None of the intent's five edge cases appear. For a measurement sprint the
  relevant ones (route skew shaping the decomposition; sub-capacity ramp windows;
  graph replay with changing route counts) directly affect whether the
  attribution is representative, so their absence is more than cosmetic.

### DoD completeness

The weakest of the three. It lists four bullet outcomes but omits the concrete
artifacts the intent's Success Criteria and Verification Strategy require: the
control artifact path, decode tok/s at the reference shape, the peer-SYS
counters, selected-token **and** generated-sequence agreement numbers, the
archived raw/parsed logs, and `STATUS.md` (Gemini updates README/STEERING/VISION
but omits the STATUS rollup the intent names). The DoD also lets the
math-vs-scaffolding hypothesis be "confirmed or refuted" without specifying the
self-consistency closure bound (sub-stages must sum to ≈ the EP total), so the
gate is softer than the intent demands.

---

## Cross-cutting gaps in *both* drafts

1. **Neither defines the reconciliation residual bound.** "Sub-stages sum to ≈
   the EP bucket" needs a number (e.g. ≤ 5% unattributed, or the run fails). Both
   leave "≈" undefined, which makes Success Criterion 1's self-consistency check
   unfalsifiable.
2. **Neither addresses how per-substage timing is captured inside the promoted
   captured graph.** This is *the* technical crux of an instrumentation sprint on
   a full-capture path, and both drafts route around it — Codex by gating on eager
   and hand-waving the graph correlation, Gemini by declaring graph authoritative
   without a mechanism. The correct answer (add fixed `cudaEventRecord` graph
   nodes behind the flag, accept that this is a separate non-promoted capture, and
   reconcile to eager) is stated by neither.
3. **Neither sets an instrumentation-overhead acceptance threshold** (flag-on vs
   flag-off tok/s delta), which is the only direct evidence that the timers aren't
   distorting the very thing they measure.
4. **Neither re-validates the Sprint 581 anchor.** Both treat 9.419 ms as a fixed
   target; if the current build no longer reproduces it (post-MTP churn), the
   reconciliation gate has no ground truth. A "first reproduce the baseline"
   step should precede decomposition.
5. **Both correctly keep MTP punted and PP dead** — no scope violation on the
   hard cuts.

## Recommendation

Use **Codex as the base** and graft three things into it:

- From Gemini: its brevity discipline — collapse Codex's repeated ABI restatement
  and clearly fence Phases 4–7 as "598+ design contract, not 597 scope."
- From this critique: add the two missing gates (reconciliation residual bound;
  instrumentation-overhead flag-on/off threshold), a "reproduce the 581 baseline
  first" step, and a concrete answer to *how* the captured-graph substages get
  measured (event nodes vs nsys mapping).
- Carry the intent's five edge cases into the 597 measurement plan, not just the
  598+ stages — at minimum record route-skew distribution so the decomposition is
  worst-case representative.

Gemini is safe to discard except as the model for how short the final 597 doc
should read once the B2-stage detail is pushed to the cycle-design appendix.

---

## Verification ledger (claims checked against source on this branch)

| Claim | Source | Verdict |
|---|---|---|
| Intent: `sync_all()` after routed FFN at line ~918 | `decode_loop.cu:918` is `sync_after_decode_stage("routed_ffn")` | **Intent wrong / Codex correct** |
| Intent: `sync_all()` at line ~1238 | `decode_loop.cu:1238` is `sync_after_decode_stage("ep_copy")` | **Intent wrong / Codex correct** |
| Unconditional `sync_all()` barriers exist in EP sequence | `decode_loop.cu:954,996,1062,1144,1170,1373` | Confirmed |
| `sync_after_decode_stage` is opt-in graph stage sync | `decode_loop.cu:210-237` (early-return unless gate+selected) | Confirmed (Codex) |
| `sync_all` = event barrier in graph mode, else per-stream sync | `decode_loop.cu:174-192` | Confirmed |
| Production GPU planner reads `d_route_totals`/`d_route_offsets_all` D2H → sets routes/active_experts/max_routes | `router_plan.cu:70-88` | Confirmed |
| `upload_post_attention_fixed_capacity_route_plan_gpu` sets routes=capacity, active=kLocalExperts, no totals readback | `router_plan.cu:198-200` | Confirmed (Codex) |
| Base `mul_mat_grouped` does synchronous D2H read of `expert_offsets[num_experts]` | `api.cc:1075-1079` | Confirmed (both) |
| `_total_tokens` variant takes host int, skips D2H | `api.cc:1173+` | Confirmed (both) |
| DS4 binds only `mmgt`/`mmgs`/`mmgs_clamped` (the `_total_tokens` variants) | `turbomind_bindings.cu:52-55` | Confirmed (Codex); Gemini omits |
| `executor_rows = rank.routes` drives GEMM | `ep_executor.cu:32,50` + `turbomind_bindings.cu:372-375` | Confirmed |
| `broadcast_ep_return_slices` loops sources, ncclBroadcast + per-dest copy, fp16 path | `runtime_pack.cu:267,325-334` | Confirmed (Codex) |
| No-SYS encoded in `PeerCopyAccounting` / `v100_nvlink_count` | `runtime_types.cuh:92,140` | Confirmed (Codex) |
| `compose_next_hidden_compact8_multi_kernel` exists | `kernels/v100/compose.cuh` | Confirmed (Codex Phase 5) |
| NCCL ReduceScatter return path | `ep_compose.cu:97` | Confirmed |
| Gemini: eager timers use `cudaDeviceSynchronize()` | eager path uses `cudaStreamSynchronize` (`decode_loop.cu:186`) | **Imprecise** |
