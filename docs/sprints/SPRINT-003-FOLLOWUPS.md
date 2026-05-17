# SPRINT-003 Follow-Ups

## 1. Run Full Real-Model Shard Emission On Persistent Scratch

**What:** Run `tools/ds4-v100-pack --emit-shards` against
`/models/DSv4-Flash-256e-fixed.gguf` with an output directory on persistent
scratch, then record shard file sizes and checksums.

**Why:** The packer dry-run and source range validation passed. The full copy
was not run because the temporary test pod uses disposable storage and the copy
would write about 145.42 GiB.

**Severity:** Important

**Suggested sprint:** Sprint 004

**Files:** `tools/ds4-v100-pack.c`, cluster scratch path docs

## 2. Add Runtime Pack Index Reader

**What:** Add runtime support for reading `pack-index.tsv` and resolving tensor
payloads from `gpuN.weights` instead of directly from GGUF offsets.

**Why:** The pack index now defines deterministic shard offsets, but `ds4.c`
still binds tensors to GGUF mmap offsets.

**Severity:** Critical

**Suggested sprint:** Sprint 004

**Files:** `ds4.c`, `ds4.h`, future pack runtime files

## 3. Wire First Source-Format GPU Upload Path

**What:** Use the pack index to upload or map source-faithful BF16/F32/F8/MXFP4
tensors into per-GPU arenas without persistent F16 dequantized copies.

**Why:** Source inspect and pack planning work. Runtime decode remains blocked
until source-format upload and kernel dispatch exist.

**Severity:** Critical

**Suggested sprint:** Sprint 004

**Files:** `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Run full real-model shard emission on persistent scratch | Important | Sprint 004 | `tools/ds4-v100-pack.c`, cluster scratch docs |
| Add runtime pack index reader | Critical | Sprint 004 | `ds4.c`, `ds4.h`, future pack runtime files |
| Wire first source-format GPU upload path | Critical | Sprint 004 | `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h` |

