# Sprint 602 - The NCCL-Free Decode Graph: Kill the Race, Unlock the Stack

Date: 2026-06-12
Status: planned

## Goal

Remove the remaining NCCL collectives from the captured decode graph —
the hc-class set Sprint 601 localized the race to — replacing them with
peer-write/kernel-reduction equivalents built on the (proven, fast,
tolerance-clean) s601 relay machinery. This single change is:

1. **The correctness fix**: the race lives inside captured NCCL
   collectives (s600) and specifically survives in the hc-class set
   (s601); the promoted path now fires token-flipping events (2/3
   controls). Removing the host kills the race.
2. **The perf unlock**: the relay+batched stack (demonstrated **208.18
   decode-domain, +23.8%**) is unpromotable only because it raises
   exposure to this race. With the race dead, it promotes.
3. **The prerequisite** for the remaining ≥50/slot levers (prefix launch
   compaction ~1.1 ms/layer, route-plan shadow ~0.45, cross-layer graph
   consolidation), which all tighten pacing further.

## The collectives to replace (per rank-layer, captured)

From the s601 Phase A/B inventory: hc 3 allreduces + hc allgather, router
allreduce, full-current broadcast, post-attention allgather (7 ops). The
EP-window set (8 EP-return broadcasts + swiglu allgather + compose
reduce-scatter) is already replaceable by the s601 `relay` + `batched`
flags. Eager/head NCCL outside the captured graph is NOT in scope (s601
a1 proved it irrelevant to the race).

## The bit-anchor decision (make it explicitly, early)

Reduction order changes bits (s600 HEAD_COMM=host measured 0.969
agreement). Policy for this sprint:

- **Allreduces (hc x3, router): ring-order-exact kernel reductions** —
  reproduce the NCCL ring's accumulation order (the promoted
  `DS4_V100_NCCL_NO_SYS_RING="0 3 2 1 5 7 6 4"` order) so outputs remain
  bit-comparable with the existing s597 control. This keeps every gate
  anchor valid and is the default plan.
- **Broadcast/allgathers (full-current, hc, post-attn): byte moves** —
  peer-write copies are bit-exact by construction (s601 relay pattern).
- Fallback if ring-order-exact proves impractical for some reduction: a
  fixed deterministic order + a CPU-reference spot-verification + a
  re-anchored control, loudly documented (VALIDATION_CONTROL_POLICY
  tolerance gate still applies).

## Plan

### Phase A - Build the kernel collective set (flag-gated)

`DS4_V100_TP_EP_HC_TRANSPORT=nccl|kernel` (default nccl): per-collective
peer-write/reduction kernels using the s601 staging/relay/event machinery
(dst^4 one-hop for SYS pairs; fixed event order; pre-allocated staging;
graph-capturable; no SYS traffic). Ring-order-exact accumulate for the
four allreduces. Per-collective bring-up with bit-verifiers (the s600
verifier pattern) at the smallest exercising shape.

### Phase B - Race execution

With `HC_TRANSPORT=kernel` + `EP_RETURN_TRANSPORT=relay` +
`SWIGLU_EXCHANGE=batched`, the captured graph contains ZERO NCCL ops.
Race gate, in escalating strictness:
1. Pairwise run-to-run checksum identity ≥ 3 x 256-step runs at the
   reference shape (the s601 detector; the old Simple-stress reproducer is
   moot with no NCCL in-graph — run it anyway as a control: it should now
   be bit-identical).
2. Event census vs history: ≥ 6 runs, zero checksum events, zero token
   events (vs promoted 1.0/run and relay 1.5/run).
3. Tolerance 1.0/1.0 vs the s597 control (valid if ring-order-exact held;
   else vs the re-anchored control per the bit-anchor policy).

### Phase C - Promote the full stack

Gates: race-zero (above), tolerance, perf ≥ +15% over the 169.01
baseline (expect ~208+ from s601's demonstration), no-SYS verification
(profiler classes + one nsys window), VRAM within budget. Flip launcher
defaults (kernel/relay/batched) with rollbacks retained. Then, if budget
remains: C-B restack, C-C route-plan shadow, prefix compaction
scoping measurement for 603.

### Phase D - Re-measure and restate the program

Reference-shape final numbers; S=1/8 stage tables; updated step-time
floor and the ≥50/slot budget + MTP-multiplier statement (s601 said
6.3-7.1x at the 123 ms floor; restate at the new floor).

## Definition of Done

1. Per-collective kernel replacements built, bit-verified, flag-gated;
   ring-order-exact policy implemented for the allreduces (or the
   fallback documented + re-anchored).
2. Zero-NCCL captured graph proven (graph node inventory shows no NCCL
   ops with the full flag stack).
3. Race verdict with event census: zero events across the gate runs, or
   the surviving signature documented (which would falsify the
   NCCL-internal hypothesis and point back at the engine).
4. Promotion per gates with launcher defaults flipped + rollbacks; or
   non-promotion evidence.
5. Final numbers + updated ≥50/slot budget + MTP-gate statement.
6. Report, follow-ups, orchestrator docs/commits.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ring-order-exact reduction is slower than NCCL LL for the hc payloads | Med | Med | Payloads are tiny (latency-bound); the s601 relay pattern beat NCCL 2.4x on the EP return; measure per-collective |
| Race survives with zero NCCL in-graph | Low-Med | High | That falsifies NCCL-internal and re-opens the engine hunt — but with a vastly smaller suspect surface (the kernel collectives are ours, instrumentable) |
| Bit-anchor breaks (ring-order-exact impractical somewhere) | Med | Med | Explicit fallback policy: deterministic order + CPU spot-verify + re-anchored control, documented |
| VRAM: staging for 7 more collective replacements | Low | Med | hc payloads are KB-scale; measure after each |
| Pacing tightens further and exposes a NEW timing bug | Med | Med | The Phase B census is the detector; jitter probes exist from s600 |

## Dependencies

- Everything through Sprint 601 committed (c962e3d2); pod environment
  intact (16 Gi shm, pack, contract, control, all opt-in flags).
- s601 relay/staging/event machinery (`engine/runtime_pack.cu`
  `ep_return_relay_graph` et al.) as the implementation template.
- s600 bit-verifier + first-divergence checksum tooling as gates.
- `DS4_V100_NCCL_NO_SYS_RING` order for ring-order-exact accumulation.
