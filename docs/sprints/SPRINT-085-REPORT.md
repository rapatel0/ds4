# Sprint 085 Report: Persistent TurboMind Sidecar Load

## Outcome

`SHIP_PERSISTENT_SIDECAR_SMOKE`.

Sprint 085 added the first executable persistent TurboMind sidecar path. The
new parser reads `turbomind-pack-index.tsv`, the CUDA smoke uploads
`gpuN.turbomind` once, rebuilds `StridedPtrH` tables from the recorded offsets
and strides, and runs TurboMind grouped MXFP4 gate/up/down from those resident
packed buffers.

This validates the format path we actually want for the appliance: offline
packing, explicit memory accounting, and no decode-time expert repacking.

## What Changed

- Added `ds4_turbomind_pack.h`.
- Added `ds4_turbomind_pack.c`.
- Added `tests/cuda_v100_turbomind_sidecar_smoke.cu`.
- Added Makefile build/clean rules for the parser and CUDA smoke.
- Recorded V100 validation in
  `logs/from-cluster/sprint085-turbomind-sidecar-v100.log`.

## V100 Evidence

Build:

```sh
CUDA_ARCH=sm_70 make tools/ds4-v100-turbomind-pack tests/cuda_v100_turbomind_sidecar_smoke
```

Pack:

```sh
./tools/ds4-v100-turbomind-pack \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /tmp/ds4-sprint085-tm-pack \
  --layer 0 \
  --kind all \
  --expert-limit 2 \
  --gpu 0 \
  --lib ./build/turbomind-v100/libggml-turbomind.so
```

Run:

```sh
./tests/cuda_v100_turbomind_sidecar_smoke \
  --lib ./build/turbomind-v100/libggml-turbomind.so \
  --tm-index /tmp/ds4-sprint085-tm-pack/turbomind-pack-index.tsv \
  --tm-dir /tmp/ds4-sprint085-tm-pack \
  --source-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --layer 0
```

Result:

```text
cuda_v100_turbomind_sidecar_smoke: layer=0 experts=2 routes=4 sidecar_bytes=26738688 max_abs=5.91128e-07 rel=0.000493098 bad=0 host_ms=0.265
cuda_v100_turbomind_sidecar_smoke: PASS
```

## Decision

Continue with the persistent sidecar path. The transient runtime bridge remains
a correctness/fallback tool, but the performance path should load offline
TurboMind packs and dispatch from resident packed buffers.

## Risks

- The smoke is bounded to layer 0 with `2/256` experts.
- Full sidecar generation can consume meaningful VRAM if source and TurboMind
  expert packs are both resident.
- Scheduler integration still needs admission control so opt-in TurboMind
  execution cannot silently overfill a 32 GB card.
- This proves the expert adapter boundary, not full-model tok/s.
