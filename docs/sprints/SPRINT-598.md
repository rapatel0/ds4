# Sprint 598 - B2-C: Eliminate SYS From the EP Return Transport

Date: 2026-06-11
Status: planned

## Goal

Replace the promoted full-capture EP return transport — 56 per-pair
`copy_f32_kernel` UVA remote loads per layer, 24 of which cross SYS at ~2 ms
each under congestion (Sprint 597: 6.92 ms/layer = 81% of the 8.52 ms EP
window) — with a no-SYS transport, behind a gate flag, validated at the
reference shape. This is the measured-first stage of the B2 cycle
(SPRINT-597-REPORT.md Phase 4 decision; design contract in
SPRINT-597-DEFERRED.md).

Target: EP return ≤ ~1 ms/layer (controls below), decode-domain throughput
≥ 1.8x the re-anchored 73.59 tok/s baseline (stretch: 2.3-2.6x ≈ 165-190
tok/s) with the tolerance gate clean.

## Controls (both must be beaten; Sprint 396 discipline)

1. **Current promoted graph copies**: 6.92 ms/layer EP return
   (s597 profiler stage table).
2. **Eager NCCL broadcast** (`broadcast_ep_return_slices`): 0.68 ms/layer
   for the same payloads (s597 reconciliation leg).

## Candidates, in order

### C1 - Capture the NCCL broadcast in-graph (try first; smallest diff)

The eager branch already moves the EP return via per-source NCCL broadcasts
on the no-SYS ring at 0.68 ms/layer (`engine/decode_loop.cu:1196-1233`,
`engine/runtime_pack.cu` `broadcast_ep_return_slices`). C1 makes the graph
branch (`opt.source_copy_schedule && opt.decode_cudagraph_gate`,
`decode_loop.cu:1174-1195`) use the NCCL broadcast path under capture
instead of the per-pair copy kernels, gated by a new flag
(e.g. `DS4_V100_TP_EP_EP_RETURN_TRANSPORT=copy|nccl`, default `copy`).

Key unknowns to resolve empirically (in a capture probe before the full
serving run): NCCL collective capture inside this 8-rank single-process
graph (group semantics, per-rank stream ordering vs the existing captured
NCCL usage), and re-capture/replay stability across steps.

### C2 - Static one-hop NVLink relay forwarding (only if C1 fails or underdelivers)

For the 12 SYS pairs, forward through one of the two NVLink relay neighbors
(relay table in `sprint597-phase01/phase1-peer-copy-analysis.txt`): fixed
staging buffers, two-stage copy kernels, fixed event order, graph-capturable,
peer copies only (no mixed NCCL+peer transport in one captured graph).
Bound ~0.2 ms/layer. Watch relay-link self-congestion (s597 follow-up #5:
SYS costs are congestion-coupled; evaluate end-to-end step time, not
per-copy means).

## Warm-up tasks (from SPRINT-597-FOLLOWUPS.md)

1. **Flag-off identity re-proof** (#3): one tolerance run of the committed
   profiler binary, flag off, vs `phase0-full-control/` — closes the s597
   deviation.
2. **Bench harness upstreaming + listen backlog** (#2): raise the
   `appliance/http_server.cu:415` listen backlog (16 → 256) and upstream the
   pod harness fixes (wave submission, UTF-8 replace, 900 s cold-load wait)
   into `tools/ds4-v100-tp-ep-http-bench.sh`, so the committed harness
   reproduces clean reference runs.

## Plan

1. Warm-up tasks (above); re-verify the s597 environment (pod, pack,
   contract, control all persist on /workspace).
2. C1 capture probe: minimal standalone or appliance-flagged run proving
   NCCL broadcast capture+replay works in the 8-rank graph (or proving it
   does not, with the failure mode recorded).
3. C1 implementation behind the transport flag; build; tolerance gate at the
   smallest exercising shape; then the reference-shape A/B (flag=copy vs
   flag=nccl) with the s597 profiler quantifying the EP-return stage.
4. If C1 < 2x decode-domain or capture-unstable: C2 relay implementation,
   same gates.
5. Reference-shape verdict: decode-domain + wall tok/s, EP-return
   stage table, no-SYS proof (profiler per-pair classes + nsys spot-check),
   tolerance vs control. Promote the winner as the default only if all
   gates pass; otherwise keep flag opt-in and record.
6. Report (SPRINT-598-REPORT.md), STATUS/steering/vision updates, commit.

## Definition of Done

1. Warm-up: identity re-proof recorded; harness fixes upstreamed; backlog
   raised (or explicitly rejected with reason).
2. A gated EP-return transport alternative exists; default unchanged until
   promotion criteria met.
3. Tolerance gate ≥ 0.99 selected-token AND generated-sequence vs the s597
   control on the winning transport.
4. EP-return stage ≤ ~1 ms/layer measured by `DS4_V100_TP_EP_EP_STAGE_PROFILE`
   on the winning transport, and no SYS-class per-pair times in the stage
   table (plus one nsys spot-check).
5. Reference-shape decode-domain ≥ 1.8x the 73.59 tok/s baseline (record
   stretch attainment vs the 2.3-2.6x projection honestly).
6. Beats BOTH controls; if promoted, the launcher default flips and the old
   path remains the rollback flag; if not promoted, the blocking evidence is
   recorded.
7. Report + STATUS/steering/VISION updates; follow-ups recorded; commits
   per repo convention (excluding `research/` and
   `VALIDATION_CONTROL_POLICY.md`).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| NCCL collectives won't capture/replay stably in the 8-rank graph | Med | Med | Dedicated capture probe before investing; C2 is the fallback; failure mode recorded either way |
| NCCL broadcast serialization grows under graph timing | Low | Med | The 0.68 ms control was measured in-process; profiler quantifies per-layer cost directly |
| Relay self-congestion (C2) erases the one-hop win | Med | Med | End-to-end step time is the gate, not per-copy means; two relay candidates per SYS pair allow load-spreading |
| HC-current (5.55 ms) caps the realized speedup below projection | High | Low | Expected; that is Sprint 599's target — record the post-C ceiling honestly |
| Backlog/harness changes shift bench comparability | Low | Low | A/B both legs with the same harness; keep the old harness path runnable |

## Dependencies

- The persistent s597 environment on gpu-01 (`/workspace/packs/...-s597`,
  `/workspace/s597-contract/`, `phase0-full-control/`), the s597 profiler
  flag, and `logs/from-cluster/sprint597-*` baselines. All present as of
  2026-06-11; the node is reserved (qwen/gemma scaled to 0).
- `SPRINT-597-REPORT.md` (decision + measured bounds),
  `SPRINT-597-DEFERRED.md` (B2-C contract), `SPRINT-597-FOLLOWUPS.md`.
