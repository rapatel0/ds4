# Sprint 578 - C1 Instability Mechanism Narrowing

Date: 2026-05-29

## Goal

Pin the exact source of the full-capture batch-instability (Sprint 577) and fix
it.

## Result summary

Narrowed by code inspection: the route-planning and compose-accumulation paths
are deterministic by construction, so they are not the source. The remaining
hypothesis is a missing intra-graph ordering/event dependency in the captured
compose region (a race that the eager path avoids with `cudaStreamSynchronize`,
which is illegal during graph capture). Not yet fixed; the fix requires pinning
the exact dependency with an ordering trace, then a rebuild/validate cycle.

## Ruled out by construction

- **Compose accumulation is deterministic on the served path.** The default
  (`compact_route_compose=true`, `compact_moe_decode_gate=true`) uses
  `ep_pack_route_dest_shards_kernel` (`kernels/v100/compose.cuh:40`), which writes
  each route to a *separate* segment (`packed[out_idx] = ...`, no atomics). The
  `atomicAdd` accumulation kernel `ep_reduce_all_dest_shards_kernel`
  (`compose.cuh:37`) is only used on the non-compact path (`decode_loop.cu:1130`),
  which the served config does not take.
- **Route ordering is deterministic.** In the compact route plan
  (`kernels/v100/router.cuh`), `route_idx = offsets_all[expert] + prior_same_expert`
  (lines 364-366) is a deterministic function of the selected experts;
  `prior_same_expert` is a deterministic count, and the `atomicAdd`s (lines 301,
  381) are order-independent counters (final counts do not depend on order). So
  `route_slots[route_idx]` placement does not race.
- **Inherent FP noise is ~0** (Sprint 576: eager-vs-eager bit-identical on matched
  tokens).

## Live hypothesis

The eager compose serializes phases with `cudaStreamSynchronize`
(`engine/ep_compose.cu:83-86, 119-122`). The captured graph cannot contain
`cudaStreamSynchronize` (illegal during capture), so the captured compose must
encode those orderings as event dependencies. If an event dependency is missing
between the cross-GPU copy/broadcast and the per-token combine (or the zero and
the accumulate), replays race. This matches every observation: graph-replay-
specific, scales with active routed tokens (more data in flight to race), eager
bit-stable, single-slot bit-exact (minimal concurrent work).

## Next steps (not completed this sprint)

1. **Ordering trace.** Run `nsys` on the captured region (lighter than
   compute-sanitizer, which OOMed in Sprint 577 - `nsys` is a tracer with far
   less memory overhead and may run on the full appliance). Look for the captured
   compose combine running concurrently with the cross-GPU copy/broadcast without
   a dependency edge. Alternatively, extend the `tp_ep_decode_top1_logit`
   diagnostic to per-stage inside the captured region.
2. **Fix.** Add the missing `cudaStreamWaitEvent`/event dependency in the captured
   compose region (or force a deterministic combine). Compile-time; rebuild.
3. **Validate.** Rerun the full-vs-full logit floor at `8`/`32` slots; the floor
   should collapse toward the eager floor (Δ -> 0). Then re-evaluate full capture
   for promotion against the eager floor.

## Status

Bug confirmed (Sprint 576), localized (Sprint 577), mechanism narrowed to a
captured-region ordering dependency (this sprint). The promoted tree is unchanged
and correct (suffix-replay default); only the `tp_ep_decode_top1_logit` diagnostic
was added. Full capture remains not promotable until the determinism defect is
fixed and validated against the eager floor.

## Definition of Done

- Deterministic sub-paths (compose accumulation, route ordering) ruled out by
  inspection, recorded.
- Live hypothesis (missing captured-region event dependency) and the `nsys`-based
  next step recorded.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.
