# Sprint 175 Follow-Ups

Date: 2026-05-22

Sprint 175 proved that `fused6_reduce` is correct and elides both
route-expanded activation staging and `down_routes`, but it was flat/slightly
slower in served 16-slot/256K A/B.

## Recommended Next Work

1. Do not promote `fused6_reduce`.
   The candidate was `70.968366` generated / `69.859485` continuation tok/s
   versus `71.349289` / `70.234456` control. This is inside run noise but below
   the promotion gate.

2. If pursuing in-GPU routed FFN next, make the next boundary larger than
   `fused6_reduce`:
   - remove or hide the `mid_half` global handoff;
   - reduce launch count across gate/up, activation, and down;
   - keep low-bit weight memory layout resident and expand only inside GPU
     tensor-core tiles or registers.

3. If the larger in-GPU boundary is too invasive, move to a broader TP/EP
   topology sprint:
   - plan a multi-layer native TP/EP group rather than a one-layer overlay;
   - require a memory planner and expected payload model before code changes;
   - use the existing routed-FFN descriptor/output-mode contract as the local
     compute primitive.

4. Keep MTP out of the throughput path until it can emit tokens without a base
   target forward for each token. Current MTP commit remains diagnostic and
   one-slot constrained.

## Evidence To Preserve

- `logs/from-cluster/sprint175-fused6-reduce/selected-token.log`
- `logs/from-cluster/sprint175-fused6-reduce/full-scheduler.log`
- `logs/from-cluster/sprint175-fused6-reduce/ab-control/summary.json`
- `logs/from-cluster/sprint175-fused6-reduce/ab-fused6-reduce/summary.json`
