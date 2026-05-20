# Sprint 085: Persistent TurboMind Sidecar Load

## Status

Complete.

## Overview

Sprint 084 created an offline TurboMind expert sidecar. Sprint 085 makes that
artifact executable in the V100 repo: the runtime can now parse
`turbomind-pack-index.tsv`, load a bounded `gpuN.turbomind` sidecar into one
device buffer, rebuild TurboMind `StridedPtrH` tables from recorded offsets,
and run the routed expert adapter without repacking weights.

This is still a bounded validation path, not the final scheduler default. The
important change is that the hot-path shape is now persistent packed weights
plus pointer tables, which is the layout the appliance needs for throughput.

## Goals

1. Add a reusable parser for `turbomind-pack-index.tsv`.
2. Reconstruct per-expert TurboMind pointer tables from sidecar offsets and
   strides.
3. Upload a bounded sidecar once into device memory.
4. Run gate/up/down grouped MXFP4 from the persistent sidecar buffer.
5. Compare output against the existing source-MXFP4 arena reference on V100.
6. Record the cluster evidence.

## Non-Goals

- Making TurboMind sidecars the default scheduler path.
- Generating full all-layer/all-expert sidecars.
- Removing source expert residency.
- Admission-controlling full sidecar memory in the planner.
- Sustained tok/s benchmarking of a full model path.

## Definition of Done

- [x] `ds4_turbomind_pack.{h,c}` parses the sidecar index schema.
- [x] `tests/cuda_v100_turbomind_sidecar_smoke` builds with `CUDA_ARCH=sm_70`.
- [x] The smoke loads a real sidecar produced by
      `tools/ds4-v100-turbomind-pack`.
- [x] The smoke runs persistent sidecar-backed TurboMind gate/up/down.
- [x] The smoke matches the source-MXFP4 arena reference on V100.
- [x] V100 log is recorded under `logs/from-cluster/`.
- [x] Artifacts are committed.

## Result

`SHIP_PERSISTENT_SIDECAR_SMOKE`.

V100 validation regenerated the layer-0, two-expert sidecar and ran the new
smoke:

```text
cuda_v100_turbomind_sidecar_smoke: layer=0 experts=2 routes=4 sidecar_bytes=26738688 max_abs=5.91128e-07 rel=0.000493098 bad=0 host_ms=0.265
cuda_v100_turbomind_sidecar_smoke: PASS
```

The measured `host_ms` is for the bounded sidecar adapter section, not full
model decode. It is useful evidence that persistent packed weights remove the
Sprint 083 transient repack tax from the adapter boundary.

## Next Step

Sprint 086 should connect this sidecar layout to memory admission and runtime
selection: generate per-GPU sidecars for an admitted layer range, account for
their bytes against 32 GB V100 limits, and let the scheduler choose persistent
TurboMind experts only when the sidecar is resident.
