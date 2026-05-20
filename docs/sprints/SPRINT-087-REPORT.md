# Sprint 087 Report: Single-Shard TurboMind Appliance Pack

## Outcome

`SHIP_APPLIANCE_PACK_BOUNDARY`.

Sprint 087 moved the TurboMind expert format into the appliance pack shape.
The project now has a CUDA packer that writes TurboMind-packed routed experts
into the same `gpuN.weights` files used by the appliance, plus a DS4 CUDA API
that executes from those prepacked resident spans without transient repacking.

## What Changed

- Added `tools/ds4-v100-appliance-pack.cu`.
- Added Makefile build/clean rules for `tools/ds4-v100-appliance-pack`.
- Relaxed `ds4_turbomind_pack.c` so `turbomind-pack-index.tsv` can point to
  `gpuN.weights`.
- Added `ds4_gpu_turbomind_mxfp4_matrix_view` and
  `ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32`.
- Extended `tests/cuda_v100_turbomind_sidecar_smoke.cu` to validate the new
  no-repack DS4 CUDA API.
- Recorded cluster validation in
  `logs/from-cluster/sprint087-appliance-pack-v100.log`.

## V100 Evidence

Build:

```sh
CUDA_ARCH=sm_70 make tools/ds4-v100-appliance-pack tests/cuda_v100_turbomind_sidecar_smoke
```

Pack bounded layer-0 experts into `gpu0.weights`:

```sh
./tools/ds4-v100-appliance-pack \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /tmp/ds4-appliance-pack-smoke \
  --layer 0 \
  --expert-limit 2 \
  --skip-non-experts \
  --pack-gpu 0 \
  --lib ./build/turbomind-v100/libggml-turbomind.so
```

Validate:

```sh
./tests/cuda_v100_turbomind_sidecar_smoke \
  --lib ./build/turbomind-v100/libggml-turbomind.so \
  --tm-index /tmp/ds4-appliance-pack-smoke/turbomind-pack-index.tsv \
  --tm-dir /tmp/ds4-appliance-pack-smoke \
  --source-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --layer 0
```

Result:

```text
cuda_v100_turbomind_sidecar_smoke: layer=0 experts=2 routes=4 sidecar_bytes=26738688 max_abs=5.91128e-07 rel=0.000493098 bad=0 host_ms=0.270
cuda_v100_turbomind_sidecar_smoke: packed_api max_abs=5.91128e-07 rel=0.000493098 bad=0
cuda_v100_turbomind_sidecar_smoke: PASS
```

## Decision

The production pack should be one appliance directory, not source shards plus
separate expert sidecars. `turbomind-pack-index.tsv` remains useful as the
expert metadata table, but its offsets point into the same per-GPU
`gpuN.weights` appliance shard.

## Risks

- Runtime scheduler binding is still pending.
- The validation is bounded to layer 0 with two experts per tensor.
- A full appliance pack can be large; full generation should run behind the
  VRAM admission report and write to persistent storage.
