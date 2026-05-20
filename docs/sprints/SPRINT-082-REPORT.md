# Sprint 082 Report: TurboMind Routed Expert Adapter Smoke

## Outcome

`SHIP_ADAPTER_SMOKE`.

Sprint 082 added the first DS4 adapter boundary for the copied TurboMind
MXFP4 grouped GEMM source. The smoke keeps DS4 source MXFP4 bytes as input,
packs them through the TurboMind C ABI, groups selected route rows by expert,
runs grouped gate/up/down GEMMs, applies DS4 clamp/SwiGLU/route-weight
semantics between GEMM phases, and compares the final routed output against the
existing DS4 source-MXFP4 arena implementation.

This is still test coverage, not the production scheduler default.

## What Changed

- Added `tests/cuda_v100_turbomind_adapter_smoke.cu`.
- Added Makefile build/link rules for the adapter smoke.
- Recorded the V100 result in
  `logs/from-cluster/sprint082-turbomind-adapter-v100.log`.

## V100 Evidence

Build:

```sh
cmake -S kernels/turbomind/ggml-turbomind -B build/turbomind-v100 \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build/turbomind-v100 --target ggml-turbomind -j 8
CUDA_ARCH=sm_70 make tests/cuda_v100_turbomind_adapter_smoke
```

Run:

```sh
./tests/cuda_v100_turbomind_adapter_smoke ./build/turbomind-v100/libggml-turbomind.so
```

Result:

```text
ds4: CUDA backend initialized on Tesla V100-SXM2-32GB (sm_70)
cuda_v100_turbomind_adapter_smoke: experts=8 routes=6 gate_kpack=0x341321 down_kpack=0x341321 max_abs=0.00129318 rel=0.000258549 bad=0
cuda_v100_turbomind_adapter_smoke: PASS
```

The test uses real DS4 matrix dimensions:

- gate/up: `N=2048,K=4096`
- down: `N=4096,K=2048`

It uses eight experts instead of the full 256-expert model shape so the smoke
is fast and VRAM-light while preserving the packed-weight, row-grouping, and
SwiGLU/down adapter contract.

## Decision

TurboMind is the preferred next runtime adapter path for routed experts because
it preserves source MXFP4 expert weights instead of expanding them to INT8 and
now matches the existing DS4 arena implementation at the routed-output boundary.

The next sprint should wire this behind an opt-in runtime flag with:

- load-time or scheduler-owned TurboMind expert packing,
- full 256-expert table construction,
- route grouping from scheduler-selected expert ids,
- fallback to the existing source-MXFP4 arena path,
- sustained V100 throughput comparison against the current default.

## Risks

- The current smoke covers eight experts, not all 256 experts.
- The TurboMind grouped ABI performs a synchronous device-to-host copy of the
  final expert offset inside each grouped call; this may matter in the hot path.
- Route sorting/grouping and full scheduler scratch ownership are not wired
  into runtime yet.
- Gate/up grouped calls were neutral to slower in Sprint 081 at very small
  token counts, so the throughput result must be measured before changing
  defaults.
