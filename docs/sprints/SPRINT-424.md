# Sprint 424: Rank-Major Routed FFN Parity Probe

## Objective

Resolve the TP/EP-only checksum divergence introduced by the default-off
`--routed-ffn-rank-major-input-gate` path.

No PP/layer-split work is in scope.

## Context

Sprint 423 proved that rank-major post-attention FFN packing can match the
slot-major path in a resident layer-2 graph and improve replay time. The same
gate improved all-layer direct decode timing, but the all-layer checksum
diverged from layer 0 onward.

That blocks HTTP promotion. The next useful work is not another broad
throughput run; it is a focused parity probe around the tensors that changed.

## Implementation

1. Re-run resident layer 0 and layer 2 A/B with the Sprint 423 gate set:
   - control: slot-major post-attention FFN input
   - candidate: rank-major routed/shared FFN input
2. If resident layer 0 matches, add focused all-layer parity instrumentation
   around:
   - device-0 slot-major `hc->d_ffn_normed`
   - rank-major shared gate/up half inputs
   - routed `r.d_a`
   - route slot maps and total route counts
3. Run all-layer direct A/B at the existing debug shape:
   - `slots=8`
   - `ctx=262144`
   - `decode_steps=4`
   - persistent graph replay on
   - deferred NCCL on
   - semantic skip stats on
4. Fix the first confirmed mismatch if it is local to the new rank-major path.

## Definition of Done

- V100 sm_70 build passes.
- Resident layer 0 and layer 2 parity are recorded.
- All-layer direct A/B is recorded after the fix or probe.
- The gate is promoted only if checksum/first-token parity holds and timing is
  not worse.
- If not promoted, the sprint records the exact mismatch location and leaves the
  gate default-off.
- `TEMP_STATUS_REPORT_424.md` and `docs/sprints/VISION.md` are updated with
  evidence.

## Outcome

Status: implemented and validated as a non-promotion.

V100 sm_70 build passed. The implementation split post-attention rank-major
scratch from HC-current rank-major scratch:

```text
RankState::d_current_full_rank_major
RankState::d_post_attn_full_rank_major
```

Resident layer parity now covers layers 0, 1, and 2:

| Case | Control checksum | Rank-major checksum | Result |
|---|---:|---:|---|
| Resident layer 0 | 4710513124 | 4710513124 | Match |
| Resident layer 1 | 2210688361 | 2210688361 | Match |
| Resident layer 2 | 4161861552 | 4161861552 | Match |

All-layer direct A/B with the dedicated buffer still diverged:

| Metric | Control | Rank-major |
|---|---:|---:|
| generated decode tok/s | 59.211511 | 63.430526 |
| continuation decode tok/s | 65.529013 | 70.936099 |
| checksum | 353694659 | 46803184 |
| first differing item | - | step 0, layer 1 |

A serial EP/dense one-step isolation also diverged, starting at layer 0, so the
blocker is not proven to be overlap-only.

Decision:

- Keep `--routed-ffn-rank-major-input-gate` default-off.
- Keep the dedicated post-attention rank-major buffer because it removes a real
  lifetime ambiguity at negligible memory cost.
- Next sprint should split shared-only vs routed-only rank-major inputs and add
  all-layer parity counters for shared gate/up half inputs, routed `r.d_a`,
  `d_next_hidden`, and `d_final_hc_shard`.

Detailed report:

```text
TEMP_STATUS_REPORT_424.md
```
