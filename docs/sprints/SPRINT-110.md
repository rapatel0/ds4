# Sprint 110 - TurboMind Gate/Up Fusion Probe

Date: 2026-05-20

## Objective

Measure whether a DS4-shaped fused TurboMind gate+up expert GEMM is materially
better on V100 than the current two-launch gate and up path.

## Context

The current appliance is correct but throughput is still about `32` aggregate
tok/s at the 8-slot/256K target. Sprint 108 showed that fusing small route
metadata work is too small to matter. Sprint 109 showed that a wider F8 row4
CTA loses to the current row2 kernel. The remaining high-signal target is the
TurboMind expert path, which is roughly a quarter of profiled GPU time and is
also where DS4 Flash's native FP4/FP8 design puts most of the model's useful
work.

Gate and up projections currently have the same routed activation input,
same expert offsets, same dtype, and same logical shape:

```text
gate: M_routes x K_hidden  *  K_hidden x N_ffn
up:   M_routes x K_hidden  *  K_hidden x N_ffn
```

A fused pack/runtime path would store gate and up as one logical expert matrix:

```text
gate_up: M_routes x K_hidden * K_hidden x (2 * N_ffn)
```

The SwiGLU step would then split the output in registers or from a contiguous
temporary. The expected win is fewer grouped-GEMM launches, one activation
read for both projections, larger N for tensor-core tiling, and less routing
table traffic. The risk is higher output bandwidth/register pressure and a
packer/runtime layout change that is not worth it if the grouped GEMM itself
does not improve.

## Plan

Add a standalone TurboMind benchmark that compares:

1. two grouped MXFP4 expert GEMM calls, one for gate and one for up;
2. one grouped MXFP4 expert GEMM call with `N = 2 * N_ffn`.

Use DS4 expert dimensions:

- `K_hidden = 4096`
- `N_ffn = 2048`
- `num_experts = 256`
- active experts: sparse six-expert route set
- tokens per active expert: `1`, `4`, and `8`

The benchmark must validate that the fused output's first half matches the
gate result and the second half matches the up result.

## Definition of Done

- Build a new `test_ggml_turbomind_grouped_gate_up_fusion` target.
- Run it on the V100 node against `libggml-turbomind.so`.
- Record separate-vs-fused timing for at least `tpa=1`, `tpa=4`, and `tpa=8`.
- If fused is materially faster, plan the production pack/runtime change:
  packed gate_up tensor, manifest kind, layer-state allocation, grouped
  runtime call, and SwiGLU split.
- If fused is neutral or slower, document why and move to a lower-level
  persistent/software-pipelined expert kernel probe.

## Decision Gate

Proceed to appliance implementation only if the fused grouped call improves the
DS4-shaped microbenchmark by at least about `5-10%` for the route counts that
matter to serving. Anything smaller is unlikely to pay for the pack/runtime
layout churn or move aggregate tok/s meaningfully.

## V100 Validation

Build target:

```text
test_ggml_turbomind_grouped_gate_up_fusion
```

The current `/workspace/ds4/build/turbomind-v100` directory only contained the
shared library, so the benchmark was compiled from the already configured
`/workspace/ds4-sprint082/build/turbomind-v100` tree and run against the current
library:

```text
/workspace/ds4/build/turbomind-v100/libggml-turbomind.so
```

The source ABI is the same `ggml_turbomind_mul_mat_grouped_total_tokens`
entrypoint used by the appliance.

## Results

| Tokens/active expert | Total routes | Separate gate+up ms | Fused gate_up ms | Speedup | Correctness |
|---:|---:|---:|---:|---:|---|
| 1 | 6 | 0.2478 | 0.1647 | 1.504x | exact |
| 4 | 24 | 0.2169 | 0.1416 | 1.532x | exact |
| 8 | 48 | 0.2082 | 0.1424 | 1.462x | exact |

All cases had `max_abs_gate=0`, `max_abs_up=0`, `rel=0`, and `bad=0`.

## Decision

Proceed to the production appliance implementation. The fused TurboMind
gate+up shape clears the decision gate by a wide margin on V100 and directly
targets the expert path that Sprint 106 measured at about a quarter of GPU
time.

The next implementation sprint should add:

- fused gate_up expert packing or a deterministic offline repack step;
- a manifest/layer-state tensor kind for the fused expert tensor;
- one grouped TurboMind call for gate+up;
- a SwiGLU split path consuming the fused output;
- a rollback knob that keeps the current two-call path available until served
  correctness and throughput pass.

## Artifacts

- `kernels/turbomind/ggml-turbomind/test_grouped_gate_up_fusion.cpp`
- `logs/from-cluster/sprint110-tm-gate-up-fusion/`
