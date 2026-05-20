# Sprint 084 Report: Offline TurboMind Expert Sidecar Pack

## Outcome

`SHIP_SIDECAR_PACKER`.

Sprint 084 added the first offline TurboMind expert packer. The tool reads the
existing V100 `pack-index.tsv`, pulls real MXFP4 expert bytes from the DS4 Flash
source GGUF, packs them through the copied TurboMind C ABI, and writes a
separate sidecar binary plus TSV index.

This keeps the normal source-layout pack index stable. The TurboMind pack is a
derived acceleration artifact, not a replacement for model provenance.

## What Changed

- Added `tools/ds4-v100-turbomind-pack.cu`.
- Added Makefile build/clean rules for `tools/ds4-v100-turbomind-pack`.
- Recorded V100 validation in
  `logs/from-cluster/sprint084-turbomind-pack-v100.log`.

## Sidecar Format

The packer emits:

- `gpuN.turbomind`
- `turbomind-pack-index.tsv`

The TSV records:

- semantic tensor id and source tensor name,
- source shape/dtype and owning GPU/layer,
- packed `N`, `K`, experts packed, experts total,
- weight/scale bytes per expert,
- TurboMind `k_pack`,
- runtime weight/scale strides,
- sidecar offsets,
- source shard file/offset/length,
- TurboMind ABI version.

Raw device pointers are not persisted. Runtime must reconstruct
`StridedPtrH[experts]` tables after uploading the sidecar to device memory.

## V100 Evidence

Build:

```sh
CUDA_ARCH=sm_70 make tools/ds4-v100-turbomind-pack
```

Run:

```sh
./tools/ds4-v100-turbomind-pack \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /tmp/ds4-sprint084-tm-pack \
  --layer 0 \
  --kind all \
  --expert-limit 2 \
  --gpu 0 \
  --lib ./build/turbomind-v100/libggml-turbomind.so
```

Result:

```text
packed blk.0.ffn_gate_exps.weight experts=2/256 N=2048 K=4096 weight=4194304 scale=262144 k_pack=0x341321
packed blk.0.ffn_up_exps.weight experts=2/256 N=2048 K=4096 weight=4194304 scale=262144 k_pack=0x341321
packed blk.0.ffn_down_exps.weight experts=2/256 N=4096 K=2048 weight=4194304 scale=262144 k_pack=0x341321
```

The bounded output was:

```text
26738688 /tmp/ds4-sprint084-tm-pack/gpu0.turbomind
1120     /tmp/ds4-sprint084-tm-pack/turbomind-pack-index.tsv
```

## Decision

Continue on the sidecar path. The transient runtime bridge from Sprint 083 is
useful as a semantic fallback and validation path, but the production path must
avoid per-token expert repacking and avoid persistent duplicate source-plus-
packed expert residency.

## Risks

- The Sprint 084 tool validates bounded output, not a full all-layer sidecar.
- The sidecar index currently records `source_checksum=pending`; before a
  production pack release, it should carry a real checksum from the source pack
  inventory.
- Runtime loading still needs separate memory accounting so TurboMind sidecars
  cannot silently overfill 32 GB V100s.
