# Sprint 083 Report: Opt-In TurboMind Runtime Routed FFN Bridge

## Outcome

`SHIP_RUNTIME_BRIDGE`.

Sprint 083 moved the copied TurboMind MXFP4 grouped GEMM from a standalone
adapter smoke into the DS4 CUDA runtime boundary. The path is opt-in and
transient: each call packs gate, up, and down expert matrices one at a time,
runs grouped TurboMind GEMMs, and frees the packed buffers.

This proves runtime semantics without changing the production default or
duplicating all expert weights in VRAM.

## What Changed

- `ds4_cuda.cu`
  - Added a `dlopen` C ABI bridge for `libggml-turbomind.so`.
  - Added device-side route counting, expert-prefix construction, route-row
    gathering, SwiGLU, and route scatter/sum for the TurboMind path.
  - Added `DS4_V100_TURBOMIND_ROUTED_FFN=1` as the opt-in selector.
  - Added `DS4_V100_TURBOMIND_STRICT=1` to fail instead of falling back.
  - Preserved fallback to the existing source-MXFP4 arena kernels.
- `tests/cuda_v100_turbomind_adapter_smoke.cu`
  - Now validates both direct TurboMind adapter output and the runtime wrapper.
- `Makefile`
  - Links CUDA binaries with `-ldl`.
  - Tracks TurboMind ABI header dependency for `ds4_cuda.o`.
- `deploy/v100/ds4-v100-appliance.env.example` and
  `tools/ds4-v100-run-appliance.sh`
  - Document and propagate the opt-in flags.

## V100 Evidence

Build:

```sh
CUDA_ARCH=sm_70 make tests/cuda_v100_turbomind_adapter_smoke tools/ds4-v100-replay
```

Run:

```sh
./tests/cuda_v100_turbomind_adapter_smoke ./build/turbomind-v100/libggml-turbomind.so
```

Result:

```text
ds4: CUDA backend initialized on Tesla V100-SXM2-32GB (sm_70)
cuda_v100_turbomind_adapter_smoke: experts=8 routes=6 gate_kpack=0x341321 down_kpack=0x341321 max_abs=0.00129318 rel=0.000258549 bad=0
cuda_v100_turbomind_adapter_smoke: runtime_wrapper max_abs=0.00129318 rel=0.000258549 bad=0 host_ms=43.298
cuda_v100_turbomind_adapter_smoke: PASS
```

The replay binary also linked successfully with the new dynamic loading path.

## Why This Is Not The Default

The bridge repacks expert weights during the routed FFN call. That is safe for
VRAM because only one expert matrix family is packed at a time, but it is not a
throughput architecture. If enabled for full-model serving, runtime packing
would dominate latency and hide the benefit of the TurboMind GEMM kernels.

The correct production direction is to eliminate runtime repack and avoid
duplicate expert residency:

- offline-convert MXFP4 expert tensors into TurboMind-ready per-GPU packs, or
- add a planner-admitted per-stage packed cache with explicit VRAM limits.

Until then, the measured production default remains the source-MXFP4 arena path
with the existing async stage pipeline and device top-1 output selection.

## Next Step

Sprint 084 should implement the offline TurboMind expert pack contract:

- extend the pack index or add a sidecar index for packed weights/scales,
- pack all 256 experts for gate/up/down per layer outside the decode loop,
- load only one expert representation by default,
- pass persistent `StridedPtrH[256]` tables into runtime,
- benchmark sustained 1M/4-slot serving against the current default.
