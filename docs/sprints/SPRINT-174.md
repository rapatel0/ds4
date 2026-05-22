# Sprint 174 - Persistent TP/EP Routed-FFN Boundary

Date: 2026-05-22
Status: Completed

## Overview

Sprint 174 moves the next practical-serving attempt from local six-route wrapper
work to a persistent tensor/expert-parallel routed-FFN boundary. Sprint 173
proved that `fused6` can remove route-expanded `a_half` and preserve
correctness, but it regressed served 16-slot/256K throughput by about `1.5%`.
That means the bottleneck is not the single activation staging buffer. The next
implementation must change how routed expert compute is scheduled across GPUs.

The target is a bounded production-shaped TP/EP executor that:

- reuses the Sprint 173 routed-FFN descriptor/output-mode shape;
- keeps TP peer ownership persistent instead of attaching copy/sync work as a
  one-layer overlay;
- validates against the current single-GPU routed-FFN output;
- measures whether a persistent boundary can improve the real served 16-slot
  256K path.

The sprint should start with a two-GPU NVLink pair because prior TP evidence is
strong enough there and the debug surface is manageable:

- Sprint 162: six-route TP proxy was `1.260x` total-with-copy on NV2.
- Sprint 163: descriptor-bound 1-token/6-route TP primitive passed correctness
  and measured `0.2071 ms -> 0.1946 ms`.
- Sprint 164/165: scheduler overlay was correct but slower because peer copies
  were bolted onto one layer rather than owned by a persistent boundary.

## Non-Goals

- No 8-way production TP rewrite in one sprint.
- No default promotion unless served A/B clears the gate.
- No further `fixed6`/`fused6` local wrapper experiments.
- No MTP changes.
- No attention/shared-FFN TP in this sprint.
- No CPU/NVMe/offload path.

## Use Cases

1. **Primitive correctness**: a two-GPU TP/EP routed-FFN boundary produces the
   same output as the current single-GPU routed FFN on real layer-3 packed
   weights.
2. **Persistent ownership**: peer arenas, peer streams, and route buffers are
   owned by a reusable executor context instead of created per call.
3. **Scheduler integration gate**: one configurable layer span can execute
   routed FFN through the persistent boundary while the rest of the runtime stays
   unchanged.
4. **Served decision**: a 16-slot/256K same-binary A/B decides whether this
   boundary should be expanded or whether the project should pivot to a larger
   in-GPU monolithic routed-FFN kernel.

## Architecture

The Sprint 174 boundary is a persistent two-rank routed executor:

```text
owner GPU
  owns layer state and final output
  owns route build
  owns TP half 0 weights
  owns peer context descriptor

peer GPU
  owns TP half 1 weights
  owns persistent x/route/weight buffers
  owns peer stream and completion event

decode call
  owner builds route metadata
  owner starts async peer payload update
  owner runs half 0
  peer runs half 1
  owner receives peer partial
  owner accumulates full output
```

The design differs from the rejected overlay in three concrete ways:

- setup is persistent at layer-open time;
- route/input buffers are reused and sized once for the admitted active slots;
- instrumentation reports copy, peer compute, owner compute, reduce, and total
  boundary time so copy/sync overhead is visible.

The first integration should be guarded by an explicit mode, for example:

```text
DS4_V100_TP_EP_ROUTED_FFN=off|layer3|span
DS4_V100_TP_EP_LAYER_FIRST=3
DS4_V100_TP_EP_LAYER_COUNT=1
DS4_V100_TP_EP_PEER=3
DS4_V100_TP_EP_VERBOSE=1
```

Default remains off.

## Implementation

### Phase 1 - Persistent Context

- Add a persistent TP/EP routed context object for one owner/peer pair.
- Allocate peer input, selected, weight, output, and scratch tensors once.
- Create peer stream/events once.
- Bind the existing TP split pack descriptors from the layer state.
- Add fail-closed validation for missing TP descriptors or invalid peer GPU.

### Phase 2 - Boundary Executor

- Implement a `ds4_tp_ep_routed_ffn_execute()` path that:
  - accepts the Sprint 173 routed descriptor shape;
  - copies or updates peer input/route payloads asynchronously;
  - runs owner half and peer half through existing TurboMind TP split views;
  - accumulates peer partial into owner output;
  - returns the same output layout as the normal routed FFN path.
- Keep `fused6` optional inside the local half only if it is already enabled;
  do not make it required.

### Phase 3 - Instrumentation

- Add timing buckets:
  - peer input/route update
  - owner half compute
  - peer half compute
  - peer output return
  - final add/reduce
  - total TP/EP boundary
- Log selected layer span, owner/peer pair, route count, active experts, and
  output-mode.

### Phase 4 - Correctness Gates

- Extend `tests/cuda_v100_tp_routed_ffn_smoke.c` or add a focused smoke that
  exercises the new persistent context.
- Validate:
  - full single-GPU output versus persistent TP/EP output;
  - `tokens=1/routes=6`;
  - `tokens=16/routes=96`;
  - accumulation mode parity.
- Run negative gates for missing TP descriptors and invalid peer selection.

### Phase 5 - Scheduler Gate

- Add a default-off scheduler/layer-executor hook for one layer or a small
  contiguous span.
- Reuse the persistent context across requests.
- Run a full selected-token smoke at 256K/16 slots with the hook enabled.

### Phase 6 - Served A/B

- Run same-binary served 16-slot/256K A/B:
  - control: production default
  - candidate: persistent TP/EP routed FFN on the configured layer/span
- Record prompt, generated, and continuation/decode tok/s separately.
- Preserve logs under `logs/from-cluster/sprint174-tp-ep-boundary/`.

## Files Summary

| File | Change |
|---|---|
| `ds4_v100_layer_execute.*` | TP/EP async-input aliasing, optional boundary timing buckets, report propagation |
| `ds4_v100_scheduler.*` | `DS4_V100_TP_EP_*` aliases and aggregate TP/EP timing fields |
| `ds4_v100_replay.c` | TP/EP timing accumulation through step-pipeline report merging |
| `tools/ds4-v100-run-appliance.sh` | deployment validation/export for `DS4_V100_TP_EP_*` flags |
| `logs/from-cluster/sprint174-tp-ep-boundary/` | V100 build, smoke, and served A/B evidence |

## Definition Of Done

- [x] Persistent owner/peer TP/EP routed context exists and defaults off.
- [x] Context validates TP descriptors, peer GPU, shapes, and buffer sizes.
- [x] Primitive smoke passes for `tokens=1/routes=6`.
- [x] Primitive smoke passes for `tokens=16/routes=96`.
- [x] Accumulation parity passes.
- [x] Negative gates fail closed.
- [x] Full selected-token smoke passes with the scheduler hook enabled.
- [x] Served 16-slot/256K A/B records prompt, generated, and continuation tok/s
      with token-match evidence.
- [x] Promote only if continuation/decode tok/s improves by at least `10%`.
- [x] If correct but slower, keep diagnostic-only and pivot to a larger
      monolithic routed-FFN kernel or scheduler topology redesign.

## Results

Implemented a default-off TP/EP naming and measurement layer over the existing
TP2 owner/peer executor:

- `DS4_V100_TP_EP_ROUTED_FFN`
- `DS4_V100_TP_EP_LAYER_FIRST`
- `DS4_V100_TP_EP_LAYER_COUNT`
- `DS4_V100_TP_EP_PEER`
- `DS4_V100_TP_EP_SHARD_DIR`
- `DS4_V100_TP_EP_ASYNC_INPUT`
- `DS4_V100_TP_EP_VERBOSE`

The runtime now records optional copy/owner/peer/copy-out/reduce/total TP/EP
boundary buckets when verbose timing is enabled. Timing stays opt-in because it
uses synchronization to make the bucket attribution meaningful.

V100 build validation:

```text
CUDA_ARCH=sm_70 make -j80 \
  ds4_v100_layer_execute.o \
  ds4_v100_scheduler.o \
  ds4_v100_replay.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_tp_routed_ffn_smoke \
  tests/cuda_v100_stage_scheduler_smoke
```

Primitive correctness on `gpu0/gpu3`:

| Shape | max_abs | rel | bad | accum_rel | total_ms | reference_ms | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `tokens=1`, `routes=6` | `9.16421e-07` | `0.000276278` | `0` | `2.23887e-08` | `0.3517` | `0.2129` | pass, slower |
| `tokens=16`, `routes=96` | `1.34401e-06` | `0.000278022` | `0` | `5.60962e-09` | `2.0604` | `2.1326` | pass, `1.035x` |

Scheduler gates:

- Stage-0 control passed at 16 slots / 256K with `tp2_layers=0`.
- TP/EP alias path passed at 16 slots / 256K with `tp2_layers=1`.
- Negative layer selection failed closed:
  `TP2 routed FFN layer 2 has no TP2 bindings`.
- Full selected-token replay passed with expected token hex `3136`.

Full selected-token verbose timing showed a cold first boundary cost, then
settled to roughly `0.39-0.47 ms` per single-slot layer-3 boundary:

```text
first boundary total_ms=33.4962
warm boundary total_ms ~= 0.3940-0.4723
```

Served same-binary 16-slot/256K A/B:

| Run | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `51.732314` | `45.984279` | `43.110262` | `16/16` |
| TP/EP layer-3 candidate | `47.701073` | `42.400954` | `39.750894` | `16/16` |

The candidate preserved correctness but regressed continuation/decode by about
`7.8%`, so it does not clear the promotion gate.

## Decision

Keep the TP/EP layer-3 path diagnostic-only. The primitive remains useful
because the 16-route-group shape is correct and slightly positive in isolation,
but the served appliance still loses when one layer is overlaid onto the current
layer-parallel scheduler. The likely issue is topology, not descriptor math:
one peer layer cannot repay the extra boundary work in the real per-step
serving loop.

Next work should pivot away from one-layer TP overlays and choose one of:

- a broader TP/EP scheduler topology where peer ownership is native over a
  layer group, not an overlay on one layer;
- a larger in-GPU routed-FFN executor that fuses gate/up, activation, down, and
  route reduction without adding inter-GPU payloads;
- a tensor-parallel prototype that changes enough layers to move the latency
  and occupancy envelope, with memory fit checked before served A/B.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Persistent TP/EP still loses to copy/sync overhead. | High | Measure copy/compute/reduce buckets separately and keep default off. |
| One-layer span is too small to show benefit. | High | Make layer span configurable, but validate one layer before widening. |
| Peer GPU is also serving its own layer stage. | High | Start with bounded diagnostics; do not promote unless served A/B proves it. |
| TP split pack availability differs from production pack. | Medium | Fail closed and use the existing Sprint 163 split pack for primitive tests. |
| Scope expands into full 8-way TP. | High | Keep Sprint 174 to one owner/peer pair and one layer/span gate. |

## Security

No new external API surface. All behavior is internal and default-off. Validate
all tensor sizes and peer ownership before launching kernels. Do not create
host-backed or persistent dequantized expert-weight copies.

## Dependencies

- Sprint 153 TP split pack format.
- Sprint 163 TP split correctness primitive.
- Sprint 164/165 scheduler overlay lessons.
- Sprint 173 routed descriptor/output-mode and liveness instrumentation.
- V100 pod with `/workspace` on local K8s storage and the existing TP split pack
  artifacts.

## Decision Gate

- **Ship/expand** only if the served 16-slot/256K continuation/decode tok/s
  improves by `>= 10%` with correctness intact.
- **Keep diagnostic** if primitive correctness passes but served A/B is flat or
  slower.
- **Pivot** if the persistent boundary cannot beat the known overlay failure
  modes or if copy/reduce buckets dominate total time.
