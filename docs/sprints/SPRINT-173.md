# Sprint 173 - Reusable Fused Routed-FFN Boundary

Date: 2026-05-22
Status: Completed

## Overview

Sprint 173 implements the next practical-serving lever for DS4 V100: a reusable
routed-FFN executor boundary for the production served shape. Sprints 154-172
have exhausted wrapper-level route build, dispatch bypass, down-only reduce,
small-route, stream, and chunking variants. The remaining work needs to change
the execution boundary itself.

The sprint starts with the current 16-slot/256K decode shape:

- `total_routes = 6`
- `active_experts = 6`
- `max_routes_per_expert = 1`
- `hidden = 4096`
- `mid = 2048`

The first implementation target is to remove the route-expanded `a_half` global
staging buffer from the candidate path. The current path writes the same token
activation row once per routed expert, then immediately reads it back for the
gate/up GEMM. Sprint 173 should keep compact MXFP4 expert weights resident,
expand/cast activations only inside the executor boundary, and return the same
full F32 output as the current path.

This is also the first TP/EP-compatible primitive. The local executor runs in
full-output mode now, but the descriptor must include an additive partial-output
mode so later tensor/expert parallel work can reuse the same boundary rather
than inventing a second execution contract.

## Non-Goals

- No broad TP/EP scheduler rewrite.
- No full 8-GPU tensor-parallel production topology.
- No MTP changes.
- No attention or shared-FFN fusion beyond optional read-only liveness notes.
- No default promotion unless the served A/B clears the throughput gate.
- No public API or generic runner work.

## Use Cases

1. Current production appliance path: `fused6` runs the served six-route routed
   FFN path in full-output mode and falls back cleanly for unsupported shapes.
2. Correctness reference: replay and focused smoke compare the candidate path
   against the current packed TurboMind path before any served benchmark.
3. TP/EP seed: a synthetic single-GPU split validates that two partial-output
   passes over disjoint route/expert subsets can reproduce one full-output pass.
4. Liveness visibility: logs report which global routed-FFN intermediates are
   still materialized and which are bypassed.

## Architecture

The sprint adds an explicit routed-FFN executor descriptor near the CUDA runtime
boundary. The descriptor should make these fields explicit:

- input layout: contiguous `x_f32` or row-pointer input
- route metadata: selected experts, route weights, sorted pairs, sorted tokens,
  expert offsets
- packed weight views: interleaved `gate_up` and `down` MXFP4 views
- ownership: full expert range now, subset fields for future EP
- output mode: `FULL_SUM_F32` now, `PARTIAL_ACCUMULATE_F32` for TP/EP

The current packed TurboMind implementation remains the fallback executor. The
candidate executor is selected only by a new opt-in mode:

```text
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6
```

Minimum candidate behavior:

```text
route build
  -> un-expanded activation staging or in-boundary activation tile
  -> TurboMind gate/up grouped MXFP4 GEMM
  -> gated-SiLU
  -> TurboMind down grouped MXFP4 GEMM
  -> weighted full-output accumulation
```

The minimum acceptable implementation may still materialize `mid_half` and
`down_routes`. It must remove or bypass the route-expanded `a_half` staging for
the supported six-route path and emit evidence that this actually happened.

## Implementation

### Phase 1 - Executor Contract

- Add a routed-FFN descriptor and output-mode enum in the CUDA/GPU API layer.
- Refactor the current packed routed path so the existing public wrappers build
  the descriptor and call the executor.
- Keep existing behavior identical when the new mode is off.
- Add `fused6` to the routed-executor selector.
- Add `fused6` to `tools/ds4-v100-run-appliance.sh` allowlist.

### Phase 2 - Six-Route Candidate

- Guard the candidate to the production shape:
  - `total_routes == 6`
  - fused interleaved gate/up active
  - gated-SiLU active
  - `hidden == 4096`
  - `mid == 2048`
- Remove route-expanded `a_half` staging.
- Preferred first slice: cast each source token row once and feed the gate/up
  GEMM through token indices instead of a route-expanded activation matrix.
- Stretch slice: add a TurboMind ABI/probe variant that reads F32 activations
  and performs F32-to-F16 conversion inside the A-tile load.
- Preserve the compact schedule, existing MXFP4 packs, and down/reduce fallback
  behavior.

### Phase 3 - Liveness And Telemetry

- Add one-shot selection logs for `fused6`.
- Emit liveness status for:
  - `a_half`
  - `gate_out`
  - `mid_half`
  - `down_routes`
  - output mode
- Keep existing timing buckets so results remain comparable to prior sprint
  profiles.

### Phase 4 - Partial-Output Smoke

- Thread `PARTIAL_ACCUMULATE_F32` through the descriptor.
- Reuse the existing accumulation behavior where possible.
- Add a synthetic single-GPU split smoke: one full pass versus two partial
  passes over disjoint route/expert subsets into a zeroed output.
- Require the partial-sum result to match the full result within the established
  TurboMind drift tolerance.

### Phase 5 - Cluster Validation

- Run `git diff --check`.
- Build affected host/CUDA code locally where possible.
- On the V100 pod:
  - build replay for `sm_70`
  - rebuild `libggml-turbomind.so` only if the stretch ABI changes it
  - run focused routed-FFN smoke
  - run direct replay with candidate off/on
  - run served 16-slot/256K same-binary A/B with prompt, generated, and
    continuation/decode tok/s split

## Files Summary

| File | Change |
|---|---|
| `ds4_cuda.cu` | Descriptor, executor dispatch, `fused6` mode, un-expanded activation path, liveness logs |
| `ds4_gpu.h` | Descriptor/output-mode declarations if a public GPU API type is needed |
| `tools/ds4-v100-run-appliance.sh` | Allow `fused6` in routed-executor launcher validation |
| `tools/ds4-v100-replay.c` | Replay/smoke coverage and selection evidence |
| `tests/cuda_v100_full_scheduler_smoke.c` | Smoke coverage with new executor mode if needed |
| `tests/cuda_v100_tp_routed_ffn_smoke.c` | Synthetic full-vs-partial additive-output validation |
| `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` | Stretch only: declare F32-activation fused ABI |
| `kernels/turbomind/ggml-turbomind/api.cc` | Stretch only: wire F32-activation fused ABI |
| `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu` | Stretch only: implement F32-to-F16 A-tile probe |

## Definition Of Done

- [x] `fused6` opt-in mode exists and defaults off.
- [x] Current default path remains behaviorally unchanged.
- [x] Routed-FFN executor descriptor exists with explicit input, route metadata,
      packed weight, ownership, and output-mode fields.
- [x] Candidate path removes/bypasses route-expanded `a_half` for the six-route
      served shape.
- [x] Logs prove `materialized_a_half=0` or equivalent for the candidate.
- [x] Replay/focused correctness passes before served benchmarking.
- [x] Synthetic full-vs-partial additive-output smoke passes.
- [x] Served 16-slot/256K A/B records prompt, generated, and continuation/decode
      tok/s separately, with token-match evidence.
- [x] Promote only if continuation/decode tok/s improves by `>= 10%` with
      correctness intact.
- [x] If correct but below `10%`, keep `fused6` opt-in and record it as the TP/EP
      primitive seed.
- [x] If no real global intermediate is removed, stop and pivot to persistent
      TP/EP boundary planning.

## Results

Implemented:

- Added `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6` in `ds4_cuda.cu`.
- Added a routed-FFN descriptor with input, route metadata, packed weight,
  ownership, and output-mode fields.
- Added the `FULL_SUM_F32` / `PARTIAL_ACCUMULATE_F32` internal output mode.
- Added a guarded six-route candidate path that uses the un-expanded activation
  route: cast one source token row once, pass `sorted_tokens` into the grouped
  gate/up GEMM, and avoid route-expanded `a_half`.
- Added liveness logging for `a_half`, `gate_out`, `mid_half`, `down_routes`,
  and output mode.
- Updated the appliance launcher allowlist to accept `fused6`.
- Extended `tests/cuda_v100_tp_routed_ffn_smoke.c` to also validate the
  `*_accum_f32` path against the full-output reference.

Validation:

- `git diff --check` passed.
- V100 build passed:
  - `CUDA_ARCH=sm_70 make -j80 ds4_cuda.o`
  - `CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-replay`
  - `CUDA_ARCH=sm_70 make -j80 tests/cuda_v100_tp_routed_ffn_smoke`
- Focused served smoke selected `fused6`, produced the expected first token
  (`3136`), and logged:

```text
ds4: TurboMind routed executor fused6 shape total_routes=6 active_experts=6 max_routes_per_expert=1
ds4: routed-FFN liveness executor=fused6 total_routes=6 route_expanded_a_half=0 compact_a_half=1 gate_out=elided mid_half=materialized down_routes=materialized output_mode=full_sum
```

- TP/partial smoke passed on the V100 pod with `fused6` enabled:
  - `tokens=1`, `routes=6`
  - `max_abs=9.16538e-07`
  - `rel=0.000276278`
  - `bad=0`
  - `accum_max_abs=2.32831e-10`
  - `accum_rel=2.4959e-08`
  - `accum_bad=0`

Same-binary served A/B at 16-slot/256K, per-step async + event handoff:

| Mode | Generated tok/s | Continuation tok/s | Prompt tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `46.227335` | `43.338126` | `52.005752` | `16/16` |
| fused6 | `45.522409` | `42.677258` | `51.212710` | `16/16` |

Evidence:

- `logs/from-cluster/sprint173-fused6-smoke/`
- `logs/from-cluster/sprint173-fused6-ab/`

## Decision

`fused6` is correct and removes a real global expanded intermediate, but it does
not improve throughput. The served A/B regressed by about `1.5%` on both
generated and continuation/decode tok/s:

- generated: `45.522409 / 46.227335 = 0.9848`
- continuation: `42.677258 / 43.338126 = 0.9848`

Keep `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6` opt-in and default off. The
result is useful as a TP/EP primitive seed because it creates the explicit
boundary and liveness accounting, but `a_half` removal alone is not the
production serving lever. Sprint 174 should move to a persistent TP/EP boundary
or a larger persistent routed-FFN boundary that removes `mid_half`/`down_routes`
and changes the GEMM scheduling, not another small wrapper around the six-route
gate/up call.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| The sprint becomes a full monolithic kernel rewrite. | High | Minimum is contract plus `a_half` removal; TurboMind ABI fusion is stretch. |
| `a_half` removal is too small to move served tok/s. | Medium | Keep promotion gate separate from architecture gate; use result to decide TP/EP next. |
| The descriptor bakes in one-GPU ownership. | High | Include ownership and partial-output fields now; validate with synthetic split smoke. |
| Indexed activation path regresses due GEMM indexing overhead. | Medium | A/B against default and preserve fallback. |
| Stretch ABI destabilizes the build. | Medium | Attempt only after Slice A correctness; rebuild `.so` and symbol-check if touched. |

## Security

No new external surface is introduced. The executor mode is internal, opt-in,
and default off. Validate route counts, expert counts, tensor sizes, ownership
ranges, and output mode before launching candidate kernels. Do not introduce
persistent dequantized expert-weight copies; source quantized and packed MXFP4
weights remain the resident formats.

## Dependencies

- Sprint 111 fused TurboMind gate/up production pack.
- Sprint 127 gated-SiLU interleaved gate/up path.
- Sprint 128 compact active-expert schedule.
- Sprint 131 indexed-A evidence.
- Sprint 163 TP partial-sum correctness evidence.
- Sprint 164/165 lesson that TP must be a persistent boundary, not a one-layer
  overlay.
- Sprint 170 follow-up requesting a true six-route fused routed-FFN executor.

## Open Questions

1. Does the un-expanded indexed activation path give enough signal, or must the
   sprint go directly to the in-kernel F32-to-F16 A-tile load?
2. What tolerance should the synthetic full-vs-partial smoke use if the existing
   TurboMind drift envelope is exceeded but selected-token correctness remains?
3. If `fused6` is correct but flat, should Sprint 174 extend this boundary into
   a persistent TP/EP prototype or attempt full gate/up + down monolithic fusion
   first?
