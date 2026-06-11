# Sprint 597 Merge Notes

Consensus run: claude-opus-4-8 (max thinking), gpt-5.5 (xhigh reasoning),
gemini-3.1-pro-preview, per the 2026-06-04 weather report
(`consensus(opus-4.8, gpt-5.5)`, Gemini as optional third). Independent drafts →
three-way cross-critique → interview → this merge.

## Draft strengths

**Claude draft (base for the final doc):**
- Correctly established the single most decision-relevant code fact: the
  promoted full-capture path already uses the fixed-capacity route plan
  (`post_attention_ffn.cu:42-45` requires `graph_event_order`;
  `router_plan.cu:198` sets `r.routes = r.route_capacity`) — so the per-layer
  host route-count readback the intent targeted is **eager-only**, and the
  promoted cost is a ~3x padded executor envelope (192 launched vs p50 64,
  Sprint 542).
- Correct TurboMind ABI story: DS4 binds only the `_total_tokens` variants
  (`turbomind_bindings.cu:52-55`); device `expert_offsets`; no per-GEMM D2H.
  No new ABI needed for worst-case-grid routing; a `_device_total_tokens`
  extension is justified only if the padded GEMM dominates the decomposition.
- Right sprint shape (597 = instrument + decide + reopen; 598+ = stages) and
  the strongest DoD of the three drafts.

**Codex draft:**
- The decisive transport correction (made in its critique, anticipated in its
  draft's caution): on the promoted graph branch
  (`source_copy_schedule && decode_cudagraph_gate`,
  `decode_loop.cu:1174-1195`), EP return is **56 per-pair
  `copy_f32_kernel` remote-load launches per layer**
  (`runtime_pack.cu:176-190` ignores device IDs); `broadcast_ep_return_slices`
  + NCCL broadcasts are the **non-graph** branch. All drafts (and the intent)
  had profiled the wrong transport.
- The no-SYS accounting gap: `record_peer_copy()` appears unused for the graph
  remote-copy kernels, so `peer_copy_sys_bytes=0` does **not** prove the
  promoted path avoids SYS on the 12 non-NVLink pairs. This became the
  interview's "SYS audit" scope decision.
- Sprint 396 correction: NCCL allreduce was **2.96x faster** than custom
  peer-doubling (4.513 ms vs 13.366 ms at 32 tokens) — the intent had it
  backwards. Custom peer transport is *not* pre-proven; the transport stage
  must beat both NCCL and the graph-copy control on evidence.
- Data-driven decision-gate branch logic (which measured dominator picks which
  598 stage); the refusal to oversell host-readback removal; line-map
  correction (`decode_loop.cu:918/1238` are opt-in `sync_after_decode_stage`,
  not `sync_all`; unconditional barriers at 954/996/1062/1144/1170/1373).

**Gemini draft:**
- Brevity discipline (model for the final doc's length).
- The "nsys on the promoted graph is the authoritative gate" framing —
  adopted in the interview.
- Static one-hop schedule over mixed NCCL/peer transport in one captured graph
  (deadlock risk), echoed by both other agents.

## Critiques accepted

- Transport model correction (Codex) — final doc profiles the graph
  `copy_f32_kernel` branch as the promoted return path; NCCL broadcast and
  ReduceScatter measured only as the non-graph/non-compact controls. **Accepted;
  verified directly in source.**
- No-SYS validation must be valid for kernel remote loads, not just
  `peer_copy_sys_bytes` / NCCL SYS edges (Codex). **Accepted** → Phase 1
  per-pair transport audit + adjacency classification.
- Sprint 396 misquote (Codex). **Accepted; verified in SPRINT-396.md** (NCCL
  4.513 ms vs doubling 13.366 ms, NCCL faster). Intent's claim inverted.
- Reconciliation residual bound; instrumentation-overhead flag-on/off
  threshold; reproduce-the-581-anchor-first step; in-graph event mechanism
  must be explicit (Claude critique's cross-cutting gaps 1-4). **Accepted** →
  DoD items 1, 3, 5; Phase 0.
- Edge cases carried into the *measurement* plan (route-skew distribution,
  sub-capacity ramp, capture hit/replay separation, zero-route ranks)
  (Claude + Codex critiques). **Accepted** → Phase 3 + DoD 8.
- Profiling-flag graph-topology mutation ("Heisenbug") and graph-cache
  invalidation risks (Gemini + Codex critiques). **Accepted** → DoD 5 reports
  node-count/cache deltas; nsys on the *unmodified* graph is the authority.
- Fence the B2 stage detail as a 598+ design contract, not 597 work
  (Claude critique of Codex's Phases 4-7). **Accepted** → design appendix +
  DEFERRED doc.

## Critiques rejected

- Gemini draft's "TurboMind ABI: No" answer (new entry point required for any
  device-resident routing). **Rejected** — conflates the base API's internal
  D2H read (not on DS4's path) with the `_total_tokens` host-scalar contract;
  promoted path already runs device-resident offsets at fixed capacity. The
  ABI extension is contingent, not required (kept as a deferred item).
- Codex draft's preference for the **eager** decomposition as the numerical
  gate. **Rejected in the interview** — the user chose nsys on the promoted
  graph as the authority (eager runs a different executor shape: ~64 actual
  routes vs the padded 192). Eager remains the reconciliation cross-check
  against the Sprint 581 anchor.
- Claude draft's embedded recommendation of the device-masked executor as the
  presumptive first 598 stage. **Softened** (Codex critique): it is one branch
  of the decision table, selected only if the padded GEMM dominates.

## Interview refinements applied

1. **Gate authority:** nsys/NVTX capture of the *unmodified* promoted
   full-capture path is the decision authority; flag-gated in-graph CUDA
   events produce the per-stage table; eager timers reconcile to the 581
   anchor.
2. **597 scope:** stage decomposition **plus** the transport ground-truth
   audit (topology dump, per-pair `copy_f32_kernel` remote-load microbench
   across all 56 directed pairs, NVLink-vs-SYS classification). The "quick
   one-hop fix probe" was declined — measurement only.
3. **README reopen:** in 597, superseding the abandonment note (MTP stays
   punted, PP stays dead).
4. **Packaging:** focused SPRINT-597 + design-contract appendix; B2 stages
   598-600 recorded in SPRINT-597-DEFERRED.md, order finalized by the
   decomposition.

## Ledger

`scripts/ledger.py` does not exist in this repo; the skill's ledger-sync step
was skipped. STATUS.md rollup remains a Sprint 597 execution deliverable.
