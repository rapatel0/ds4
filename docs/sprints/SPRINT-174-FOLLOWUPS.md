# Sprint 174 Follow-Ups

Date: 2026-05-22

Sprint 174 kept the one-layer TP/EP path diagnostic-only. Correctness is good,
but the served 16-slot/256K candidate regressed continuation throughput by about
`7.8%`.

## Recommended Next Work

1. Plan a broader topology change instead of another one-layer overlay.
   The isolated 16-token TP primitive is slightly positive, but one layer cannot
   repay the peer boundary cost inside the current layer-parallel serving loop.

2. Choose between two real levers:
   - a native TP/EP scheduler group over multiple layers, with peer ownership
     resident for the group;
   - a larger in-GPU routed-FFN executor that fuses gate/up, activation, down,
     and route reduction without inter-GPU payloads.

3. If pursuing TP/EP next, require a memory planner before implementation:
   - layer group span;
   - owner/peer assignment;
   - TP split pack availability;
   - per-GPU VRAM delta;
   - expected boundary payload per token and per active slot.

4. If pursuing the in-GPU routed executor next, target the whole routed FFN, not
   another wrapper:
   - packed activation staging;
   - gate/up MXFP4;
   - gated SiLU;
   - down MXFP4;
   - route-weighted accumulation.

## Evidence To Preserve

- `logs/from-cluster/sprint174-tp-ep-boundary/full-selected-token.log`
- `logs/from-cluster/sprint174-tp-ep-boundary/full-selected-token.json`
- `logs/from-cluster/sprint174-tp-ep-boundary/ab-control/summary.json`
- `logs/from-cluster/sprint174-tp-ep-boundary/ab-tp-ep/summary.json`

