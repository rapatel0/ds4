# Sprint 081 Report: Copied TurboMind MXFP4 Grouped GEMM Proof

## Outcome

`SHIP_PROOF`.

Sprint 081 copied the TurboMind C ABI wrapper plus the required lmdeploy
`turbomind` support tree into `ds4`, patched the copied CMake default so it
uses the copied tree, built `libggml-turbomind.so` on V100, and ran the
grouped MXFP4 compare smoke on DS4 gate/up and down expert shapes.

This is the first copied-source proof that matches DS4's routed expert source
format. It is ready for a hot-path adapter sprint.

## Changes

- Added copied TurboMind source under `kernels/turbomind/`:
  - `ggml-turbomind/` C ABI wrapper and tests;
  - `lmdeploy/src/turbomind/` support source;
  - `lmdeploy/LICENSE`;
  - `README.md`.
- Patched `kernels/turbomind/ggml-turbomind/CMakeLists.txt` so the default
  `LMDEPLOY_SRC` is `../lmdeploy/src`, not the old deepseek path.
- Extended copied `test_grouped_compare.cpp` with CUDA event timing for grouped
  vs six single-expert calls.
- Updated architecture and vision docs with the copied-source kernel decision.

## Validation

V100 configure:

```bash
cmake -S kernels/turbomind/ggml-turbomind \
  -B build/turbomind-v100 \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_BUILD_TYPE=Release
```

V100 build:

```bash
cmake --build build/turbomind-v100 \
  --target ggml-turbomind test_ggml_turbomind_grouped_compare -j 8
```

V100 smoke:

```bash
./build/turbomind-v100/test_ggml_turbomind_grouped_compare \
  ./build/turbomind-v100/libggml-turbomind.so
```

The node was idle at measurement start: all eight V100s reported `0 MiB`
memory and `0%` GPU utilization.

## Results

| Case | Shape | Active experts | Tokens/expert | Grouped ms | Six single calls ms | Speedup | Max abs | Rel | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `down_decode` | `N=4096 K=2048` | 6 | 1 | `0.1037` | `0.1307` | `1.261x` | `3.1250e-02` | `2.3798e-07` | PASS |
| `gate_up_decode` | `N=2048 K=4096` | 6 | 1 | `0.1454` | `0.1294` | `0.890x` | `6.2500e-02` | `5.6460e-07` | PASS |
| `down_prompt` | `N=4096 K=2048` | 6 | 4 | `0.1057` | `0.1305` | `1.234x` | `6.2500e-02` | `2.6239e-07` | PASS |
| `gate_up_prompt` | `N=2048 K=4096` | 6 | 4 | `0.1315` | `0.1305` | `0.992x` | `6.2500e-02` | `4.8738e-07` | PASS |

## Decision

Plan the next sprint around a DS4 TurboMind routed-expert adapter.

The copied grouped MXFP4 path is correct for the DS4 expert shapes and removes
the need to convert source MXFP4 experts to INT8. The microtiming is mixed:
TurboMind grouped down is faster than six single-expert calls, while grouped
gate/up is roughly equal or slightly slower at very small token counts. That is
still the better integration target because the real scheduler can batch across
slots/routes and because TurboMind keeps the model's native expert format.

The next sprint should not attempt a full serving rewrite. It should add the
adapter boundary:

- pack one layer's source MXFP4 routed experts into TurboMind layout at load;
- gather active route rows into TurboMind's expert-offset contract;
- run grouped gate/up GEMMs;
- apply SwiGLU and route weights;
- run grouped down GEMM;
- scatter/sum back into the existing routed output buffer;
- compare against the current source-MXFP4 routed FFN output.

## Dependency Caveat

The copied TurboMind source is local. The current CMake still uses
`FetchContent` for header/build dependencies such as fmt and CUTLASS. That is
acceptable for this proof, but a production appliance build should either cache
or copy those build dependencies as well.

## Artifacts

- `logs/from-cluster/sprint081-turbomind-v100.log`
