# Sprint 235 - Descriptor Backed Full-Layer TP/EP Scaffold

Date: 2026-05-23
Status: Planned

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

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] New tool is TP/EP-only and does not modify PP scheduler files.
- [ ] New tool parses layer-2 dense/control/expert/KV/comp descriptors from
      the real TP/EP contract.
- [ ] New tool copies and device-checks real dense/control descriptor bytes on
      the owning V100s.
- [ ] New tool runs the existing descriptor-backed TurboMind EP expert path.
- [ ] New tool runs the sharded KV gate at `32` slots / `256K`.
- [ ] V100 run reports per-family rows and bytes for all eight GPUs.
- [ ] V100 run passes deterministic finite checks:
      `kv_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint235-tp-ep-full-layer-scaffold/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

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

Pending.
