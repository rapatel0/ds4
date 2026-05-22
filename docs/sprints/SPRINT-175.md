# Sprint 175 - Fused Six-Route Routed-FFN Reduce Boundary

Date: 2026-05-22
Status: Completed

## Overview

Sprint 175 returns to the in-GPU routed-FFN boundary after Sprint 174 showed
that a one-layer TP/EP overlay is correct but slower in served 16-slot/256K
mode. The target is not another isolated gate/up or down probe. It is a larger
six-route executor mode for the production serving shape that removes both
route-expanded activation staging and route-expanded down-output staging.

The current production-shaped served path exposes:

- `total_routes = 6`
- `active_experts = 6`
- `max_routes_per_expert = 1`
- `hidden = 4096`
- `mid = 2048`
- fused interleaved MXFP4 gate/up
- gated-SiLU enabled

Sprint 173 proved `fused6` can bypass route-expanded `a_half`, but it still
materialized `mid_half` and `down_routes`. Sprint 171 proved the six-route
down-reduce epilogue is correct, but it was not useful alone. This sprint joins
those two pieces into one explicit executor mode:

```text
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce
```

The mode remains default-off. It must fail closed if the six-route down-reduce
ABI is missing, and it must emit liveness evidence that `down_routes` is elided.

## Non-Goals

- No default promotion without a served A/B win.
- No 8-way tensor-parallel scheduler rewrite.
- No MTP changes.
- No attempt to remove `mid_half` in this sprint; that requires a true
  cross-GEMM fused kernel or persistent in-kernel handoff.
- No change to model format, pack format, or default appliance configuration.

## Use Cases

1. A production-shaped 16-slot/256K served request can select
   `fused6_reduce` and preserve token correctness.
2. Verbose logs prove the candidate elides both route-expanded `a_half` and
   `down_routes` while keeping `mid_half` materialized.
3. The current generic path, `fixed6`, `fixed96`, `fixed768`, and `fused6`
   remain behaviorally unchanged.
4. Same-binary served A/B decides whether combining the two prior primitives
   is a material throughput lever.

## Architecture

The candidate is a bounded executor composition:

```text
route build
  -> compact source activation cast
  -> grouped MXFP4 gate/up + gated-SiLU
  -> mid_half
  -> grouped MXFP4 down with route-weighted F32 reduce epilogue
  -> out_f32
```

The important change is the execution contract, not a new model dtype. Packed
MXFP4 weights stay resident. Activations are converted to FP16 only for the
Volta tensor-core GEMM boundary. The down projection writes F32 accumulated
token rows directly through the existing TurboMind reduce epilogue, avoiding
the global `down_routes` buffer and the follow-on scatter/reduce kernel.

## Implementation

### Phase 1 - Runtime Selector

- Add `DS4_CUDA_TM_ROUTED_EXECUTOR_FUSED6_REDUCE`.
- Accept aliases such as `fused6_reduce`, `indexed6_reduce`, and `reduce6`.
- Add launcher allowlist support.
- Keep the mode default-off.

### Phase 2 - Scratch And Liveness

- Reuse the Sprint 173 unexpanded activation path.
- Require the six-route down-reduce ABI for `fused6_reduce`.
- Elide `down_routes` scratch only when the candidate can actually use the
  reduce epilogue.
- Update liveness logs so the candidate reports `down_routes=elided`.

### Phase 3 - Down-Reduce Dispatch

- Allow `fused6_reduce` to force the existing six-route down-reduce epilogue
  without requiring the broader `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE`
  diagnostic flag.
- Fail closed if the forced epilogue is unavailable.
- Preserve the existing env-gated down-reduce diagnostic behavior for other
  modes and shapes.

### Phase 4 - Validation

- Run `git diff --check`.
- Build affected local targets.
- On the V100 pod:
  - build `ds4_cuda.o` and `tools/ds4-v100-replay`;
  - run selected-token/full-scheduler smoke with `fused6_reduce`;
  - confirm logs show `route_expanded_a_half=0` and `down_routes=elided`;
  - run same-binary served 16-slot/256K A/B against default control.

## Files Summary

| File | Change |
|---|---|
| `ds4_cuda.cu` | Add `fused6_reduce`, force six-route down-reduce epilogue, elide `down_routes` scratch |
| `tools/ds4-v100-run-appliance.sh` | Allow `fused6_reduce` in deployment validation |
| `deploy/v100/ds4-v100-appliance.env.example` | Document the new opt-in mode |
| `logs/from-cluster/sprint175-fused6-reduce/` | V100 build, smoke, and served A/B evidence |

## Definition Of Done

- [x] `fused6_reduce` opt-in mode exists and defaults off.
- [x] Launcher accepts the new mode.
- [x] Default path and prior routed-executor modes remain unchanged.
- [x] Candidate elides route-expanded `a_half` and `down_routes` in liveness
      logs.
- [x] Candidate fails closed if the six-route down-reduce epilogue is missing.
- [x] V100 build passes for affected targets.
- [x] V100 selected-token/full-scheduler smoke passes with the candidate.
- [x] Served 16-slot/256K A/B records prompt, generated, and continuation
      tok/s separately with `16/16` token match.
- [x] Promote only if continuation/decode tok/s improves by at least `10%`.
- [x] If correct but below the gate, keep diagnostic-only and decide whether
      the next sprint should remove `mid_half` with a true fused kernel or move
      to broader TP/EP topology.

## Results

Implemented:

- Added `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce` in `ds4_cuda.cu`.
- Added launcher allowlist aliases:
  `fused6_reduce`, `fused_6_reduce`, `ffn_fused6_reduce`,
  `unexpanded6_reduce`, `indexed6_reduce`, and `reduce6`.
- Forced the existing six-route TurboMind down-reduce epilogue for this mode
  without requiring the broader diagnostic
  `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` flag.
- Elided `down_routes` scratch only when the six-route reduce ABI is present.
- Made the candidate fail closed if `fused6_reduce` is requested but the
  down-reduce path cannot be selected.
- Documented the new opt-in mode in the V100 appliance env example.

Validation:

- `git diff --check` passed.
- Local Mac CUDA build is not meaningful because the Makefile has no CUDA
  compiler configured there.
- V100 build passed on `llm/llamacpp-build-8gpu`:

```text
make ds4_cuda.o \
  tools/ds4-v100-replay \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke \
  CUDA_ARCH=sm_70 -j80
```

Selected-token smoke passed on the V100 pod with expected token `3136` and
logged the required liveness evidence:

```text
ds4: TurboMind routed executor fused6_reduce shape total_routes=6 active_experts=6 max_routes_per_expert=1
ds4: routed-FFN liveness executor=fused6_reduce total_routes=6 route_expanded_a_half=0 compact_a_half=1 gate_out=elided mid_half=materialized down_routes=elided output_mode=full_sum
ds4: TurboMind down-reduce epilogue selected total_routes=6
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
```

Full 16-slot/256K scheduler smoke passed:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

Same-binary served A/B at 16-slot/256K, 16 requests x 64 generated tokens,
per-step async + event handoff:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|---:|
| control | `20.066988` | `71.349289` | `70.234456` | `16/16` |
| fused6_reduce | `19.959853` | `70.968366` | `69.859485` | `16/16` |

The candidate preserved correctness and removed a real global intermediate, but
it was slightly slower: about `-0.53%` generated and `-0.53%` continuation.
That does not clear the `>= 10%` promotion gate.

Evidence:

- `logs/from-cluster/sprint175-fused6-reduce/selected-token.log`
- `logs/from-cluster/sprint175-fused6-reduce/full-scheduler.log`
- `logs/from-cluster/sprint175-fused6-reduce/ab-control/summary.json`
- `logs/from-cluster/sprint175-fused6-reduce/ab-fused6-reduce/summary.json`

## Decision

Keep `fused6_reduce` diagnostic-only and default-off.

Combining unexpanded activation staging with down-route reduce is not the
missing throughput lever. The remaining materialized `mid_half` boundary and
the two separate grouped GEMM launches are now the most plausible in-GPU
routed-FFN bottleneck. The next larger in-GPU attempt should either remove the
`mid_half` global handoff with a true persistent/fused routed-FFN executor, or
move back to broader TP/EP topology where dense work is natively scheduled
across devices instead of overlaid on one layer.
