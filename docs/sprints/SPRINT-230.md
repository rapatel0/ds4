# Sprint 230 - TP Dense And KV Slice

Date: 2026-05-23
Status: Planned

## Overview

Sprint 229 proved that the separate TP runtime can own all eight GPUs and
allocate target hidden/KV/scratch arenas. Sprint 230 adds the first bounded
dense/KV slice inside that runtime.

The purpose is not to implement a real DS4 attention layer yet. The purpose is
to make sharded KV ownership concrete: per-layer offsets, per-slot rows,
ratio-4/indexer handling, resident hidden input, device-side KV update, and
device-side KV read/verify.

## Goals

- Extend `ds4_v100_tp_runtime` with explicit per-layer KV shard layout.
- Add a TP-only dense/KV fixture API.
- Update a bounded layer/slot/position row from resident hidden state.
- Support both:
  - ratio-4 layer with indexer KV;
  - ratio-128 layer without indexer KV.
- Verify KV row contents on all eight GPUs.
- Report row offsets, row bytes, and max error.

## Non-Goals

- No PP scheduler changes.
- No generic scheduler abstraction.
- No real DS4 attention math.
- No real low-bit weight loading.
- No EP routed expert work.
- No MTP.
- No throughput claim.

## Implementation

1. Add per-layer TP KV layout to `ds4_v100_tp_runtime.cu`.
2. Add `ds4_v100_tp_runtime_dense_kv_slice` to the TP runtime API.
3. Extend `tools/ds4-v100-tp-runtime-smoke.cu` with:
   - `--dense-kv-slice`;
   - `--layer N`;
   - `--slot N`;
   - `--position N`;
   - `--indexer on|off`.
4. The slice should:
   - fill resident hidden input deterministically;
   - compute a deterministic shard row on-device;
   - write attn KV row at the correct layer/slot/position offset;
   - optionally write indexer KV row for ratio-4 layers;
   - read the row back and verify expected values.
5. Build locally enough to check non-CUDA guard and formatting.
6. Build/run on the V100 pod for:
   - target allocation `32` slots / `256K`;
   - layer `2`, ratio `4`, indexer on;
   - layer `3`, ratio `128`, indexer off.
7. Copy evidence to
   `logs/from-cluster/sprint230-tp-dense-kv-slice/`.

## Files In Scope

| File | Purpose |
|---|---|
| `ds4_v100_tp_runtime.h` | dense/KV slice API |
| `ds4_v100_tp_runtime.cu` | TP KV layout and device kernels |
| `tools/ds4-v100-tp-runtime-smoke.cu` | dense/KV slice smoke options |
| `docs/sprints/SPRINT-230.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint230-tp-dense-kv-slice/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Runtime has explicit per-layer KV shard layout.
- [ ] Dense/KV fixture API is exposed in TP-only runtime files.
- [ ] Smoke supports `--dense-kv-slice`.
- [ ] V100 ratio-4 layer slice passes with indexer KV enabled.
- [ ] V100 ratio-128 layer slice passes with indexer KV disabled.
- [ ] Smoke reports offsets, row bytes, and max error.
- [ ] No PP scheduler files are modified.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint230-tp-dense-kv-slice/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Decision

Pending.
