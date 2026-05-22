# TEMP Status Report 003

Date: 2026-05-21

## Current Topline

The current best practical served baseline remains the production gated
TurboMind appliance at 16 slots / 256K context with per-step async pipeline and
event handoff:

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---:|
| production baseline | 256K | 16 | `46.316374` | `43.421600` | `16/16` |
| graph replay opt-in | 256K | 16 | `40.058341` | `37.554695` | `16/16` |
| synchronous 16-slot batch | 256K | 16 | `12.948545` | `12.139261` | `16/16` |
| single-slot baseline | 256K | 1 | `4.440185` | `13.907372` | fixture matched |
| single-slot MTP commit | 256K | 1 | `4.373406` | `13.539950` | fixture matched |

Average GPU utilization in the 16-slot served graph A/B stayed low, roughly
`5-10%` by `nvidia-smi` samples, so the practical problem remains launch,
scheduling, and small-shape execution rather than raw VRAM fit.

## Sprint 169 Result

Commit: `90e1cf7 sprint 169 explicit graph stream`

The CUDA Graph blocker from Sprint 157 is fixed. The TurboMind routed-FFN graph
path now captures on explicit nonblocking per-GPU streams instead of the legacy
default stream.

Evidence:

- V100 build passed with `make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay`.
- Graph-on replay and served appliance correctness matched graph-off.
- Graph-on server log recorded `43` captured graph keys.
- There were `0` begin-capture failures and `0` graph launch failures.

Decision: keep `DS4_V100_TURBOMIND_GRAPH=1` diagnostic-only. It improves a
direct replay continuation micro-measurement but regresses the actual served
16-slot/256K path.

## New Scouting Checks

The per-step async pipeline is still the production winner because it overlaps
the layer-sharded stages. Its cost is that each stage worker calls the layer
executor with `n_slots=1`, so the routed FFN sees the six-route per-request
shape.

The synchronous batch path gives the routed FFN the dense 96-route shape, but
it serializes the stages and drops served throughput to `12.948545` generated
tok/s. That makes "just batch more slots" the wrong next move unless we also
preserve stage overlap.

MTP commit mode is not a performance feature yet. On the short fixture it
accepted `8/15` draft attempts, but the implementation still runs the target
forward for every generated token and only records whether the MTP draft would
have matched. It measured slightly slower than base single-slot decode.

## Current Interpretation

The next material implementation should target the shape that the best serving
path actually uses:

```text
per stage, per slot, per routed layer:
  total_routes = 6
  active_experts = 6
  max_routes_per_expert = 1
```

The previous 96/768/1536-route specialized kernels are useful diagnostics, but
they do not help the current best served topology unless we redesign the
scheduler boundary.

## Best Next Sprint Candidate

Build a persistent or fused DS4 routed-FFN executor for the six-route per-slot
shape used by the per-step async pipeline. The target is not another wrapper
graph. It should reduce host launch overhead and keep the MXFP4 gate/up,
activation, down, and weighted reduce boundary together for the current
production execution shape.

Fallback if that does not move served throughput: restart TP/EP work as a
broader persistent topology boundary, not the previous copy-heavy per-layer
TP2 overlay.
