# Sprint 174 - Persistent TP/EP Routed-FFN Boundary

Date: 2026-05-22
Status: Planned

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
| `ds4_cuda.cu` | TP/EP executor context, owner/peer execution, timing/logging |
| `ds4_v100_layer_state.*` | expose TP split descriptors and peer metadata as needed |
| `ds4_v100_layer_execute.*` | guarded layer hook for configured TP/EP span |
| `ds4_v100_scheduler.*` | pass TP/EP mode/config into layer execution if needed |
| `tests/cuda_v100_tp_routed_ffn_smoke.c` | persistent-context correctness and timing smoke |
| `tools/ds4-v100-run-appliance.sh` | allowlist/export TP/EP runtime flags |
| `tools/ds4-v100-replay.c` | log selected TP/EP mode in replay/server evidence if needed |

## Definition Of Done

- [ ] Persistent owner/peer TP/EP routed context exists and defaults off.
- [ ] Context validates TP descriptors, peer GPU, shapes, and buffer sizes.
- [ ] Primitive smoke passes for `tokens=1/routes=6`.
- [ ] Primitive smoke passes for `tokens=16/routes=96`.
- [ ] Accumulation parity passes.
- [ ] Negative gates fail closed.
- [ ] Full selected-token smoke passes with the scheduler hook enabled.
- [ ] Served 16-slot/256K A/B records prompt, generated, and continuation tok/s
      with token-match evidence.
- [ ] Promote only if continuation/decode tok/s improves by at least `10%`.
- [ ] If correct but slower, keep diagnostic-only and pivot to a larger
      monolithic routed-FFN kernel or scheduler topology redesign.

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
