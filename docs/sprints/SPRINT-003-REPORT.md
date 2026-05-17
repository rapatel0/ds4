# SPRINT-003 Report: Manifest-Driven Packer Baseline

Date: 2026-05-17

## Verdict

`SHIP` for the packer baseline: the Sprint 002 manifest can now be consumed to
produce deterministic per-GPU shard offsets, validate source GGUF byte ranges,
write a pack index, and optionally emit `gpuN.weights` shard files.

Runtime decode is still blocked by the missing V100 source-format upload and
kernel dispatch path.

## Implemented

- Added `tools/ds4-v100-pack.c`.
- Added `make tools/ds4-v100-pack`.
- Added dry-run shard planning from `SPRINT-002-PACK-MANIFEST.tsv`.
- Added source GGUF range validation when `--source` is supplied.
- Added deterministic per-GPU shard offsets with configurable alignment
  (`--align`, default `256` bytes).
- Added `--write-index` to emit `pack-index.tsv`.
- Added explicit `--emit-shards` mode to copy model bytes into `gpuN.weights`.
  The tool does not copy tensor payloads by default.

## Artifacts

- Cluster dry-run log:
  `docs/sprints/drafts/SPRINT-003-PACK-DRYRUN.log`
- Derived pack index:
  `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`

## Shard Plan

Using the measured source manifest and 256-byte shard alignment:

| GPU | Tensors | Payload | Padded shard size |
|---:|---:|---:|---:|
| gpu0 | 173 | 20.98 GiB | 20.98 GiB |
| gpu1 | 186 | 20.02 GiB | 20.02 GiB |
| gpu2 | 186 | 20.02 GiB | 20.02 GiB |
| gpu3 | 186 | 20.02 GiB | 20.02 GiB |
| gpu4 | 186 | 20.02 GiB | 20.02 GiB |
| gpu5 | 158 | 16.69 GiB | 16.69 GiB |
| gpu6 | 152 | 16.67 GiB | 16.67 GiB |
| gpu7 | 101 | 11.01 GiB | 11.01 GiB |
| total | 1328 | 145.42 GiB | 145.42 GiB |

Total padding from 256-byte alignment is about `0.032 MiB`.

## Validation

Local:

```bash
make tools/ds4-v100-pack
./tools/ds4-v100-pack \
  --manifest docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv
./tools/ds4-v100-pack \
  --manifest docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv \
  --out-dir /tmp/ds4-pack-index \
  --write-index
```

Also ran a tiny `--emit-shards` smoke test with a synthetic source file and
manifest to verify byte copying and shard offsets.

Cluster:

```bash
/tmp/ds4-v100-pack \
  --manifest docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /tmp/ds4-pack-index \
  --write-index
```

This validated all source offsets against the real 145.42 GiB GGUF without
copying model bytes.

## Not Done

- Full `--emit-shards` against the real model has not been run yet; it would
  write about 145.42 GiB of shard files and should use persistent scratch, not
  the temporary pod filesystem.
- Runtime code does not yet load `pack-index.tsv` or `gpuN.weights`.
- V100 source FP8/MXFP4 upload and kernel dispatch remain the next blocking
  implementation step.

