# Sprint 084: Offline TurboMind Expert Sidecar Pack

## Status

Complete.

## Overview

Sprint 083 proved an opt-in TurboMind runtime bridge, but that bridge repacks
expert weights during the routed FFN call. Sprint 084 starts the production
format path: a separate TurboMind expert sidecar/index that can be built
offline from the source DS4 GGUF and normal `pack-index.tsv`.

The sidecar is intentionally separate from `pack-index.tsv`. The existing pack
index remains the source-layout contract; the TurboMind index is a derived
acceleration artifact that records packed weight/scale offsets and the ABI
metadata needed to reconstruct runtime pointer tables.

## Goals

1. Add a CUDA conversion tool for TurboMind MXFP4 expert sidecars.
2. Read real source tensor offsets from the existing V100 pack index.
3. Pack `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` through copied
   TurboMind.
4. Emit a per-GPU sidecar binary plus TSV index.
5. Support `--expert-limit` for bounded cluster validation.
6. Validate the tool on V100 against the real DS4 Flash source GGUF.

## Non-Goals

- Loading the sidecar in the runtime scheduler.
- Replacing the default source-MXFP4 arena path.
- Building the full all-layer/all-expert sidecar in this sprint.
- Persisting raw device pointer tables in the sidecar.

## Definition of Done

- [x] `tools/ds4-v100-turbomind-pack` builds with `CUDA_ARCH=sm_70`.
- [x] The tool reads `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`.
- [x] The tool reads real source bytes from `/models/DSv4-Flash-256e-fixed.gguf`.
- [x] The tool writes `gpuN.turbomind` and `turbomind-pack-index.tsv`.
- [x] The index records packed byte sizes, `k_pack`, strides, source shard
      offsets, and TurboMind ABI version.
- [x] V100 log is recorded under `logs/from-cluster/`.
- [x] Artifacts are committed.

## Result

`SHIP_SIDECAR_PACKER`.

Cluster validation packed layer 0 gate/up/down with `--expert-limit 2`:

```text
packed blk.0.ffn_gate_exps.weight experts=2/256 N=2048 K=4096 weight=4194304 scale=262144 k_pack=0x341321
packed blk.0.ffn_up_exps.weight experts=2/256 N=2048 K=4096 weight=4194304 scale=262144 k_pack=0x341321
packed blk.0.ffn_down_exps.weight experts=2/256 N=4096 K=2048 weight=4194304 scale=262144 k_pack=0x341321
```

The bounded sidecar size was `26,738,688` bytes. The full production sidecar
should be generated per GPU and admitted by the memory planner before runtime
loads it.

## Next Step

Sprint 085 should add a loader for `turbomind-pack-index.tsv`, upload a bounded
sidecar into device memory, reconstruct `StridedPtrH` tables after upload, and
run the existing adapter smoke from persistent sidecar buffers instead of
runtime repacking.
