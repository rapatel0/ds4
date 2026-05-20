# Sprint 087: Single-Shard TurboMind Appliance Pack

## Status

Complete.

## Overview

Sprint 087 pivots the TurboMind work from sidecar scaffolding to the production
appliance shape. The new packer emits a single appliance directory with
per-GPU `gpuN.weights` files. Non-expert tensors remain described by
`pack-index.tsv`; routed expert tensors are stored in the same `gpuN.weights`
files in TurboMind MXFP4 grouped layout and described by
`turbomind-pack-index.tsv`.

This preserves the one-appliance-pack contract while keeping the expert
metadata separate enough for the runtime to choose the correct execution path.

## Goals

1. Add an appliance packer that writes TurboMind-packed routed experts into
   `gpuN.weights`, not `gpuN.turbomind`.
2. Keep non-expert tensors in the normal `pack-index.tsv`.
3. Keep expert rows in `turbomind-pack-index.tsv` with offsets into the same
   per-GPU appliance shard.
4. Add a no-repack DS4 CUDA API that consumes already-packed TurboMind expert
   spans from a resident arena.
5. Validate the packer and CUDA API on V100 against the source-MXFP4 arena
   reference.

## Non-Goals

- Full all-layer/all-expert appliance pack generation.
- Scheduler default selection of the TurboMind appliance pack.
- Removing the old source-MXFP4 fallback path.
- Full model tok/s benchmarking.

## Definition of Done

- [x] `tools/ds4-v100-appliance-pack` builds with `CUDA_ARCH=sm_70`.
- [x] The packer emits `gpuN.weights`, `pack-index.tsv`, and
      `turbomind-pack-index.tsv`.
- [x] `turbomind-pack-index.tsv` can point to `gpuN.weights`.
- [x] `ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32` runs from
      prepacked resident spans without repacking.
- [x] V100 smoke validates direct TurboMind and the DS4 CUDA no-repack API
      against the source-MXFP4 arena reference.
- [x] V100 log is recorded under `logs/from-cluster/`.
- [x] Artifacts are committed.

## Result

`SHIP_APPLIANCE_PACK_BOUNDARY`.

Bounded V100 validation generated an appliance-shaped pack with layer-0
gate/up/down experts, two experts each, inside `/tmp/ds4-appliance-pack-smoke`:

```text
ds4-v100-appliance-pack: gpu0.weights bytes=26738688
```

The persistent adapter smoke then loaded `gpu0.weights` through
`turbomind-pack-index.tsv` and validated both the direct TurboMind call and the
new DS4 CUDA no-repack API:

```text
cuda_v100_turbomind_sidecar_smoke: layer=0 experts=2 routes=4 sidecar_bytes=26738688 max_abs=5.91128e-07 rel=0.000493098 bad=0 host_ms=0.270
cuda_v100_turbomind_sidecar_smoke: packed_api max_abs=5.91128e-07 rel=0.000493098 bad=0
cuda_v100_turbomind_sidecar_smoke: PASS
```

## Next Step

Wire the scheduler to open an appliance directory: load non-expert rows from
`pack-index.tsv`, load routed expert rows from `turbomind-pack-index.tsv`,
allocate arena bytes from the combined extents, and dispatch FFN through the
no-repack TurboMind API when the layer has TurboMind expert bindings.
