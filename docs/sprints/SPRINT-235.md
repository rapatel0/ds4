# Sprint 235 - Descriptor Backed Full-Layer TP/EP Scaffold

Date: 2026-05-23
Status: Complete

## Overview

Sprint 234 proved that the separate TP/EP path can read real TurboMind
production-pack expert bytes and execute the layer-2 routed expert slice on all
eight V100s. Sprint 235 expands that from expert-only byte binding into a
full-layer descriptor scaffold for layer `2`: dense TP rows, replicated
control/router rows, sharded KV rows, and EP experts must all be represented,
loaded or touched on the owning GPU, and reported from one TP/EP-only tool.

This sprint is still not serving and does not claim logits equivalence. It is
the bridge from descriptor ownership plus expert execution to a full-layer
runtime shape that can later be replaced row by row with true DS4 math.

## Goals

- Keep the hard cut: no PP scheduler edits and no generic PP/TP abstraction.
- Add new TP/EP-only code for descriptor-backed full-layer scaffolding.
- Parse the Sprint 228 TP/EP contract for one layer.
- Parse the production TurboMind index for routed expert byte offsets.
- For layer `2`, bind every descriptor family:
  - dense TP rows;
  - replicated control/router rows;
  - EP expert rows;
  - KV shard rows;
  - compression-state rows.
- Copy/touch real dense and control descriptor bytes on their owning V100s.
- Preserve the Sprint 234 real TurboMind expert execution path.
- Preserve the Sprint 230 sharded KV correctness gate at `32` slots / `256K`.
- Emit per-GPU byte counts for dense, control, EP expert, KV, and compression
  state.
- Emit deterministic device-side checksums for loaded dense/control bytes so
  we prove residency and GPU access, not just host parsing.
- Run the full-layer scaffold on the V100 pod at:
  - `32` slots;
  - `256K` context;
  - `top_k=6`;
  - layer `2`;
  - MTP off.

## Non-Goals

- No PP/layer-split work.
- No changes to `ds4_v100_scheduler.*`.
- No generic scheduler shared by PP and TP.
- No full logits-equivalent DS4 layer claim.
- No serving integration.
- No MTP.
- No all-43-layer loop yet.
- No dense FP8/BF16 matmul correctness claim yet; this sprint proves
  descriptor-backed residency and execution scaffolding for the full layer.

## Architecture

Add a new TP/EP-only full-layer smoke instead of extending the PP path:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
  parse tp-ep-pack-contract.tsv
  parse turbomind-pack-index.tsv
  open ds4_v100_tp_runtime
  load/touch dense TP rows on owning GPU
  load/touch replicated control rows on each GPU
  run sharded KV slice
  run descriptor-backed TurboMind EP experts
  report per-family bytes, timings, and deterministic checksums
```

The smoke is intentionally separate from `tools/ds4-v100-tp-ep-layer-smoke.cu`
so the TP/EP path can grow without inheriting fixture assumptions. It may reuse
small helper patterns from Sprint 234, but it should not introduce a PP/TP
shared scheduler layer.

Dense/control rows use source bytes from the current production pack sidecars.
The device check is a bounded checksum kernel over copied row bytes. This is a
scaffold check, not final math. The next sprint will replace selected dense
checksum stages with real low-bit dense kernels and compare layer outputs.

## Implementation

1. Add `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
2. Add a Makefile target for the new tool under the CUDA-only section.
3. Implement a tool-local contract parser for:
   - `dense_tp`;
   - `replicated_control`;
   - `ep_expert`;
   - `kv_shard`;
   - `kv_comp_state`.
4. Implement a sidecar byte loader that:
   - joins `--pack-dir` with `source_pack_file`;
   - reads `source_shard_offset` and `bytes_estimate` for dense/control rows;
   - copies bytes to the row owner GPU;
   - computes a device checksum and byte count.
5. Reuse the Sprint 234 TurboMind index parser/loader pattern for routed
   expert bytes.
6. Reuse the separate TP runtime and `ds4_v100_tp_runtime_dense_kv_slice` for
   the KV gate.
7. Print a single summary line with:
   - slots/context/top_k/layer;
   - per-family descriptor rows;
   - per-family loaded bytes;
   - descriptor checksum;
   - KV max_abs;
   - worst EP ms;
   - full-layer scaffold ms;
   - repeat_bad/repeat_nan;
   - PASS/FAIL.
8. Build and run on the V100 pod against:
   - `/workspace/logs/sprint228-tp-ep-pack-contract/contract/tp-ep-pack-contract.tsv`;
   - `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
9. Copy evidence to
   `logs/from-cluster/sprint235-tp-ep-full-layer-scaffold/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | new full-layer TP/EP scaffold |
| `Makefile` | CUDA target for the new tool |
| `docs/sprints/SPRINT-235.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint235-tp-ep-full-layer-scaffold/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] New tool is TP/EP-only and does not modify PP scheduler files.
- [x] New tool parses layer-2 dense/control/expert/KV/comp descriptors from
      the real TP/EP contract.
- [x] New tool copies and device-checks real dense/control descriptor bytes on
      the owning V100s.
- [x] New tool runs the existing descriptor-backed TurboMind EP expert path.
- [x] New tool runs the sharded KV gate at `32` slots / `256K`.
- [x] V100 run reports per-family rows and bytes for all eight GPUs.
- [x] V100 run passes deterministic finite checks:
      `kv_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`.
- [x] Evidence is copied to
      `logs/from-cluster/sprint235-tp-ep-full-layer-scaffold/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack: `/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint228-tp-ep-pack-contract/contract/tp-ep-pack-contract.tsv`

Command shape:

- `32` slots;
- `256K` context;
- `top_k=6`;
- layer `2`;
- MTP off.

Full-layer scaffold result:

| Metric | Value |
|---|---:|
| total layer rows | `288` |
| dense rows | `112` |
| control rows | `136` |
| expert rows | `16` |
| KV rows | `16` |
| comp rows | `8` |
| dense loaded bytes | `163102720` |
| control loaded bytes | `84041408` |
| EP loaded bytes | `641728512` |
| descriptor checksum | `3434523335` |
| aggregate routes | `192` |
| dispatch bytes | `1572864` |
| return bytes | `1572864` |
| route imbalance | `1.000000` |
| runtime bytes/GPU | `7122628608` |
| KV max_abs | `0.000000000` |
| descriptor load/check ms | `2414.124867` |
| dense/KV ms | `0.744619` |
| worst gate ms | `0.164284` |
| worst down ms | `0.085094` |
| worst EP ms | `0.249378` |
| scaffold ms | `2415.118864` |
| repeat max_abs | `0.000000000` |
| repeat bad/nan | `0 / 0` |
| result | `PASS` |

Per GPU, the tool reports `14` dense rows, `17` control rows, `2` expert
rows, `2` KV rows, and `1` comp row for layer `2`.

## Risks

- The current TP/EP contract points dense/control source bytes at the current
  production pack layout. Some rows are logical TP shards over a source span,
  not a physically repacked TP shard. This is acceptable for Sprint 235 because
  the sprint proves byte binding and device residency, not final dense math.
- Reading the same source sidecar for all TP ranks is not the final residency
  format. A later packer sprint should emit TP-native sidecars once the runtime
  row contract stabilizes.
- Device checksum kernels can prove residency and GPU access, but they do not
  prove DS4 numerical correctness.
- Rank-7 expert timing skew from Sprints 231-234 may remain visible.

## Decision

Complete. The new TP/EP-only full-layer scaffold parses the real TP/EP
contract, binds all layer-2 descriptor families, device-checks real
dense/control bytes, runs the sharded KV gate, and preserves the
descriptor-backed TurboMind EP execution path on all eight V100s.

This is still not serving and not a logits-equivalent DS4 layer. The
`descriptor_ms` is dominated by host-side one-shot loading/checking and is not
a runtime throughput metric. The next sprint should replace the dense/control
checksum scaffold with real descriptor-backed low-bit dense execution for a
representative full layer, while keeping MTP off.
