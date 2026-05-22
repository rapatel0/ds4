# Sprint 173: Reusable Fused Routed-FFN Boundary

## Overview

Sprint 173 should stop spending effort on wrapper-only route-build, dispatch,
and tail-kernel tweaks. The current packed TurboMind routed path in
`cuda_tm_routed_mxfp4_packed_impl()` still materializes a multi-step execution
chain around the real hot boundary:

1. route build into `offsets` / `sorted_pairs` / `sorted_weights`
2. activation staging into `a_half`
3. gate/up grouped MXFP4 execution
4. gated-SiLU or standalone SwiGLU
5. down grouped MXFP4 execution
6. route-weighted reduction into final `[token, hidden]`

Sprints 126, 170, 171, and 172 jointly show that step 1 alone, fixed-shape
dispatch bypass, and down-only epilogues are not the missing lever. The next
sprint should introduce a reusable routed-FFN executor contract that keeps the
route metadata explicit, keeps compact weight traffic in MXFP4, expands to
FP16/FP32 only inside the executor boundary, and removes at least one expanded
global activation staging buffer from the candidate path.

The bounded first slice is the existing six-route served decode shape:

- `total_routes = 6`
- `active_experts = 6`
- `max_routes_per_expert = 1`
- `hidden = 4096`
- `mid = 2048`

This sprint is still an implementation sprint, not an analysis sprint, but the
implementation should be deliberately narrow:

- define the reusable executor descriptor first;
- add a local full-output mode now;
- reserve a partial-output mode in the API now so TP/EP does not need a second
  incompatible boundary later;
- prototype the six-route path by eliminating route-expanded `a_half` staging
  inside the candidate executor path;
- keep the current packed TurboMind path as the fallback and promotion control.

## Use Cases

1. **Current one-GPU production path**: the layer executor can call a routed
   FFN boundary that still returns full `[tokens, hidden]` F32 output, but no
   longer requires route-expanded `a_half` global staging in the candidate
   path.
2. **Future TP/EP reuse**: a later TP/EP path can call the same boundary with
   the same route metadata and packed weight views, but request partial additive
   output instead of final full output.
3. **Explicit fallback safety**: the runtime can decline unsupported shapes and
   fall back to the current `gate_up/down + scatter/reduce` chain without
   changing correctness behavior or appliance defaults.
4. **Liveness accounting**: profiling/log output can state exactly which routed
   intermediates were materialized for a call (`a_half`, `gate_out`, `mid_half`,
   `down_routes`) so future sprints can continue from measured facts rather than
   assumptions.
5. **Bounded six-route promotion gate**: the production 16-slot/256K appliance
   can A/B the new boundary behind an explicit opt-in flag without entangling
   TP2 overlay logic, scheduler topology, or broader route-builder changes.

## Architecture

Sprint 173 should add a DS4-owned routed executor descriptor at the GPU API
layer and make the current packed path one implementation of that contract.

```text
router select
    |
    v
route build
  sorted_pairs / sorted_weights / offsets
    |
    v
ds4 routed executor descriptor
  - input layout
  - route metadata layout
  - packed weight views
  - output mode
    |
    +--> fallback executor
    |     current a_half -> gate/up -> mid_half -> down_routes -> reduce/scatter
    |
    +--> fused candidate executor
          in-boundary activation tile -> gate/up -> gated-SiLU -> down
          -> weighted accumulation
```

### Executor Contract

The new contract should be explicit about four things that are currently spread
across scratch layout, env flags, and helper conventions:

- **Input layout**
  - `x_f32` contiguous batch input, or
  - row-pointer table input for batch-by-pointer call sites
- **Route metadata layout**
  - `expert_offsets[num_groups + 1]`
  - `sorted_pairs[total_routes]`
  - `sorted_tokens[total_routes]` when token indirection is used
  - `sorted_weights[total_routes]`
- **Packed weight layout**
  - fused `gate_up` view or separate `gate` / `up` views
  - `down` view
  - compact active-expert schedule, when present
- **Output mode**
  - `FULL_SUM_F32`: current one-GPU behavior, writes final `[tokens, hidden]`
  - `PARTIAL_SUM_F32`: additive partial result for future TP/EP merge

The contract should live below `ds4_v100_layer_execute.c`, but be shaped so the
layer executor can keep calling it in `FULL_SUM_F32` mode now and switch to
`PARTIAL_SUM_F32` later without another API rewrite.

### First Fused Boundary

The first fused boundary should keep route build outside the kernel boundary and
fuse the post-route work:

- activation formation
- gate/up grouped execution
- gated-SiLU
- down grouped execution
- route-weighted accumulation

The first concrete implementation target is not “full persistence everywhere.”
It is:

- **remove route-expanded `a_half` global staging from the candidate path**
- keep packed MXFP4 expert weights unchanged
- keep compact schedule semantics unchanged
- preserve a clean fallback to current grouped TurboMind calls

That means the candidate executor should read token activations from the
original token buffer, form FP16 activation tiles inside the boundary, and feed
those tiles directly into the exact six-route gate/up + down sequence without
writing the old `[total_routes, hidden]` `a_half` buffer to global memory.

### TP/EP Compatibility

Sprint 164/165 proved the old TP2 overlay failed because copy/sync work was
bolted onto one layer after the fact. Sprint 173 should avoid repeating that
mistake by defining the routed boundary so that:

- the executor does not assume one GPU always owns the final output;
- output can be additive and mergeable;
- route metadata is caller-owned and reusable across local and TP/EP modes;
- packed weight views remain valid for both full and split ownership.

The sprint does not need to integrate TP/EP scheduling now. It does need to
leave a usable seam for it.

## Implementation

### Phase 1: Define The Routed Executor Descriptor

**Files**

- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_v100_layer_execute.h`

**Tasks**

- Add a DS4-owned routed executor descriptor and mode enum at the GPU API
  boundary.
- Encode input mode, route metadata pointers, packed weight views, and output
  mode explicitly instead of relying on scratch conventions inside
  `cuda_tm_routed_mxfp4_packed_impl()`.
- Keep the initial layer-executor call sites on `FULL_SUM_F32` only.
- Reserve `PARTIAL_SUM_F32` in the API and thread it through validation and
  logging even if the first sprint does not execute a TP/EP mode.

### Phase 2: Refactor The Current Path Behind The Contract

**Files**

- `ds4_cuda.cu`
- `ds4_gpu.h`

**Tasks**

- Move the existing packed routed FFN flow behind the new descriptor as the
  fallback implementation.
- Keep support for:
  - fused gate/up vs separate gate + up
  - compact schedule
  - existing route-row reduce / down-reduce epilogue fallbacks
  - row-pointer batch entry points
- Preserve current public entry points such as:
  - `ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32`
  - `..._accum_f32`
  - `..._batch_ptr_table_f32`
  by making them construct the descriptor rather than own the full execution
  choreography directly.

### Phase 3: Add The Six-Route Fused Candidate

**Files**

- `ds4_cuda.cu`
- `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`
- `kernels/turbomind/ggml-turbomind/api.cc`
- `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu`

**Tasks**

- Add a new guarded routed-executor candidate for the exact six-route decode
  shape.
- Preferred shape:
  - `total_routes == 6`
  - `num_experts == 6`
  - fused interleaved gate/up
  - gated-SiLU active
  - `hidden == 4096`
  - `mid == 2048`
- Implement the candidate so it does not materialize route-expanded `a_half` in
  global memory.
- Keep `mid_half` and `down` accumulation bounded inside the new boundary even
  if they are still global for the first sprint.
- If a true monolithic kernel is too large, the minimum acceptable candidate is
  a chained executor path that still removes `a_half` and keeps the remaining
  steps under one executor selection/telemetry surface.
- Select the candidate behind a new opt-in routed-executor mode and keep
  default behavior unchanged.

### Phase 4: Add Liveness And Selection Instrumentation

**Files**

- `ds4_cuda.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-replay.c`

**Tasks**

- Emit one clear selection log when the fused candidate is active.
- Extend routed profiling/logging to report:
  - `materialized_a_half`
  - `materialized_gate_out`
  - `materialized_mid_half`
  - `materialized_down_routes`
  - output mode (`full` vs `partial`)
- Keep the existing stage timing buckets so A/B data still compares cleanly to
  Sprint 126 and later runs.
- Make replay and appliance logs show enough evidence to prove whether the new
  boundary really removed `a_half`.

### Phase 5: Verification And Promotion Gate

**Files**

- `tests/cuda_v100_full_scheduler_smoke.c`
- `tests/cuda_v100_tp_routed_ffn_smoke.c`
- `docs/sprints/VISION.md`
- `logs/from-cluster/sprint173-fused-routed-boundary/` (artifact output)

**Tasks**

- Add a correctness smoke for the new executor path on the real six-route
  shape.
- Extend the TP routed smoke so it can validate that the new contract preserves
  additive-output semantics, even if the sprint only exercises local full mode.
- Run direct replay on the real model with candidate off and on.
- Run 16-slot/256K served A/B with prompt, generated, and continuation tok/s
  split recorded.
- Promote only if continuation/decode tok/s improves by at least `10%` with
  correctness intact.
- If correctness passes but the gain is below `10%`, keep the path opt-in and
  treat the contract as the TP/EP seed instead of a production default.
- If the candidate cannot remove a real global intermediate, stop and pivot
  explicitly to TP/EP boundary work next.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `ds4_gpu.h` | Modify | Define the routed executor descriptor, input/output modes, and reusable GPU API surface. |
| `ds4_cuda.cu` | Modify | Refactor the current packed routed FFN behind the descriptor, add the six-route fused candidate, and add liveness instrumentation. |
| `ds4_v100_layer_execute.h` | Modify | Reserve a stable layer-executor-facing mode/contract shape so future TP/EP callers do not need a second API. |
| `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` | Modify | Add any bounded DS4/V100 routed-executor ABI needed for the six-route candidate. |
| `kernels/turbomind/ggml-turbomind/api.cc` | Modify | Wire the new ABI through TurboMind state/workspace handling. |
| `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu` | Modify | Implement the six-route candidate entry point if the fused boundary requires a new probe kernel. |
| `tools/ds4-v100-run-appliance.sh` | Modify | Expose the new opt-in executor mode and preserve current allowlist safety. |
| `tools/ds4-v100-replay.c` | Modify | Add explicit replay coverage and selection evidence for the new boundary. |
| `tests/cuda_v100_full_scheduler_smoke.c` | Modify | Cover scheduler correctness with the new executor mode enabled. |
| `tests/cuda_v100_tp_routed_ffn_smoke.c` | Modify | Assert the contract remains compatible with additive partial-output semantics. |
| `docs/sprints/VISION.md` | Modify after execution | Record the result and the next decision. |

## Definition of Done

- [ ] A routed-FFN executor descriptor exists and is used by the current packed
      routed FFN wrappers instead of open-coded scratch choreography.
- [ ] The descriptor explicitly supports `FULL_SUM_F32` now and reserves
      `PARTIAL_SUM_F32` for future TP/EP work.
- [ ] A guarded six-route candidate path exists for the production decode shape.
- [ ] The candidate path removes route-expanded `a_half` global staging, with
      log or profiler evidence proving `materialized_a_half=0`.
- [ ] Replay and full-scheduler smokes pass with the new mode off and on.
- [ ] TP routed smoke still passes or is updated to validate additive-output
      compatibility for the new contract.
- [ ] 16-slot/256K served A/B records generated, prompt, and continuation tok/s
      separately.
- [ ] Defaults remain unchanged unless continuation/decode tok/s improves by at
      least `10%` with correctness intact.
- [ ] If the path is correct but below the promotion bar, it remains opt-in and
      is explicitly documented as the TP/EP primitive seed rather than a failed
      dead end.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The sprint drifts into a full persistent scheduler rewrite instead of a bounded executor boundary. | Medium | High | Keep route build outside the first fused boundary and scope the implementation to the six-route post-route path. |
| The candidate only renames the path without actually removing meaningful global staging. | Medium | High | Require explicit liveness counters and a DoD item for `a_half` elimination. |
| The new contract bakes in one-GPU full-output assumptions and blocks later TP/EP reuse. | Medium | High | Reserve partial additive output mode in the API now, even if only full mode executes this sprint. |
| The monolithic kernel surface is too large for one sprint. | High | Medium | Accept a chained executor implementation only if it still removes `a_half` and lands the reusable contract. |
| Instrumentation becomes noisy or incomparable with existing routed profile history. | Medium | Medium | Add liveness fields alongside existing timing buckets instead of replacing them. |
| The six-route candidate helps the exact served shape but is unusable for denser future shapes. | Medium | Medium | Keep the contract shape generic and treat the six-route kernel as one implementation of that contract, not the contract itself. |

## Security

- Keep all new behavior local to CUDA/runtime entry points and existing replay or
  appliance tooling; do not expose new executor modes through network APIs.
- Validate route counts, expert counts, tensor sizes, and output mode before the
  candidate path launches; fail closed to the existing implementation.
- Do not introduce persistent dequantized weight buffers or secondary expanded
  global weight copies.
- Keep source quantized bytes and appliance-packed MXFP4 weights as the only
  persistent expert-weight residency format.

## Dependencies

- Sprint 111 fused `gate_up` production pack and runtime path.
- Sprint 127 gated-SiLU interleaved fused gate/up path.
- Sprint 128 compact active-expert scheduling.
- Sprint 131 indexed-A result as the already-tested “wrapper-only” baseline that
  this sprint should move beyond.
- Sprint 163 TP routed smoke and Sprint 164/165 overlay lessons for future
  additive partial-output semantics.
- Existing TurboMind copied tree under `kernels/turbomind/`.
- V100 `sm_70` build and replay environment for final execution.

## Open Questions

1. Should the first contract expose separate input modes for contiguous token
   batches and row-pointer batches, or normalize both to one internal form
   before executor selection?
2. Is the minimum acceptable first candidate a single new kernel, or is a
   chained executor acceptable as long as `a_half` staging is gone and the
   contract is real?
3. Should `PARTIAL_SUM_F32` mean “partial hidden contribution for TP split”,
   “partial expert ownership contribution for EP”, or both under one additive
   contract?
4. If the six-route path is correct but only mildly positive, does Sprint 174
   extend the same contract to a denser synthetic route shape first, or pivot
   immediately to TP/EP using the new boundary?
