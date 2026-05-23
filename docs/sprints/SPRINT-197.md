# Sprint 197 - Routed FFN Liveness Profile Contract

Date: 2026-05-23
Status: Completed

## Objective

Make the remaining routed-FFN materialization boundary measurable in the real
runtime, so the next monolithic/persistent kernel sprint has a concrete target:
remove or reduce `mid_half` materialization rather than tuning another wrapper.

## Context

Sprints 195-196 showed that direct TP4 collectives are not a decode-serving win
for the current 16-slot/256K payload. The production-serving branch therefore
pivots back to routed FFN execution.

The current TurboMind routed wrapper already combines gate/up, SwiGLU, down,
and sum at the C ABI level. The remaining boundary is inside the CUDA/TurboMind
path:

- `a_half`: activation staging, either route-expanded or compact token rows;
- `mid_half`: gated-SiLU route activations consumed by down GEMM;
- `down_routes`: down-projection route outputs, already elided by
  `fused6_reduce` when available.

Before attempting a deeper kernel rewrite, the runtime should report these
materializations under `DS4_V100_TURBOMIND_PROFILE=1`.

## Scope

- Extend TurboMind profile stats in `ds4_cuda.cu` with liveness counters:
  - scratch bytes;
  - `a_half` bytes;
  - `mid_half` bytes;
  - `down_routes` bytes;
  - calls with route-expanded activation staging;
  - calls with compact activation staging;
  - calls with materialized `mid_half`;
  - calls with materialized `down_routes`;
  - calls using down-reduce epilogue.
- Print these fields in the existing `turbomind_profile` summary.
- Preserve current defaults and performance behavior.
- Validate with a V100 build and a profile-enabled smoke.

## Non-Goals

- No new fused kernel in this sprint.
- No promotion of `fused6_reduce`.
- No served-throughput claim from instrumentation alone.

## Definition Of Done

- [x] Profile summary includes the new liveness counters.
- [x] The default runtime behavior is unchanged when profiling is off.
- [x] V100 build passes.
- [x] A profile-enabled V100 smoke emits the new fields.
- [x] Sprint/vision/status artifacts are updated with the measured result.
- [x] Changes are committed.

## Implementation

`DS4_V100_TURBOMIND_PROFILE=1` now reports:

- `route_expanded_a_calls`
- `compact_a_calls`
- `mid_half_calls`
- `down_routes_calls`
- `down_reduce_epilogue_calls`
- `scratch_bytes`
- `a_half_bytes`
- `mid_half_bytes`
- `down_routes_bytes`
- `avg_scratch_bytes`
- `avg_mid_half_bytes`

These counters live in the existing profile path, so normal serving behavior is
unchanged when `DS4_V100_TURBOMIND_PROFILE` is off.

## Validation

Static check:

```text
$ git diff --check
```

V100 build:

```text
$ make -j80 CUDA_ARCH=sm_70 ds4_cuda.o
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -arch=sm_70 ...

$ make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -arch=sm_70 ...
```

Profile smoke:

```text
DS4_V100_TURBOMIND_PROFILE=1
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_FUSED_GATE_UP=1
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce
DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE=1
./tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --synthetic-prompt-token 926 \
  --synthetic-prompt-len 8 \
  --ctx 262144 \
  --tokens 1 \
  --json
```

Result:

```text
rc=0
ds4: routed-FFN liveness executor=fused6_reduce total_routes=6
     route_expanded_a_half=0 compact_a_half=1 gate_out=elided
     mid_half=materialized down_routes=elided output_mode=full_sum
```

Representative GPU0 profile:

```text
calls=48
route_expanded_a_calls=0
compact_a_calls=48
mid_half_calls=48
down_routes_calls=0
down_reduce_epilogue_calls=48
scratch_bytes=1746432
a_half_bytes=393216
mid_half_bytes=1179648
down_routes_bytes=0
avg_scratch_bytes=36384.0
avg_mid_half_bytes=24576.0
gate_up_pct=46.65
down_pct=24.44
```

## Decision

The current exact production-shaped fused six-route path has already removed
two avoidable materializations:

- route-expanded activation rows are replaced by compact token rows;
- `down_routes` is elided by the down-reduce epilogue.

The remaining materialized routed boundary is `mid_half`, and it appears on
every profiled routed FFN call. For the six-route decode shape it is only
`24 KiB` per call, so the next sprint should not expect a large bandwidth win
from merely shrinking that buffer. The real next implementation must reduce
launch/GEMM boundary cost or change the execution shape, for example by:

- introducing a true persistent gate/up plus down executor; or
- fusing the down projection into the gate/up kernel at the tile level; or
- shifting back to larger batched/prefill TP4 shapes where the new doubling
  collective can be amortized.
