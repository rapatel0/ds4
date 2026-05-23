# Sprint 230 - TP Dense And KV Slice

Date: 2026-05-23
Status: Complete

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

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Runtime has explicit per-layer KV shard layout.
- [x] Dense/KV fixture API is exposed in TP-only runtime files.
- [x] Smoke supports `--dense-kv-slice`.
- [x] V100 ratio-4 layer slice passes with indexer KV enabled.
- [x] V100 ratio-128 layer slice passes with indexer KV disabled.
- [x] Smoke reports offsets, row bytes, and max error.
- [x] No PP scheduler files are modified.
- [x] Evidence is copied to
      `logs/from-cluster/sprint230-tp-dense-kv-slice/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

Built on the V100 pod:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4-sprint181 &&
   make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke'
```

Target allocation smoke:

```text
./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 32 --kv-dtype f8 --scratch-mib 1536
```

Result:

```text
tp_runtime_smoke ctx=262144 slots=32 hidden=4096 scratch_bytes=1610612736 fixture_max_abs=0.000000000
gpu 0 hidden_bytes 524288 kv_bytes 3707940864 comp_state_bytes 1803550720 scratch_bytes 1610612736 total_bytes 7122628608
...
gpu 7 hidden_bytes 524288 kv_bytes 3707940864 comp_state_bytes 1803550720 scratch_bytes 1610612736 total_bytes 7122628608
```

Ratio-4 layer with indexer KV:

```text
./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 32 --kv-dtype f8 --scratch-mib 1536 \
  --dense-kv-slice --layer 2 --slot 7 --position 1024 --indexer on
```

Result:

```text
tp_dense_kv_slice ctx=262144 slots=32 hidden=4096 layer=2 ratio=4 slot=7 position=1024 attn_row=384 indexer_row=256 max_abs=0.000000000
gpu 0 attn_offset 811153664 attn_row_bytes 65 indexer_offset 815401216 indexer_row_bytes 17
...
gpu 7 attn_offset 811153664 attn_row_bytes 65 indexer_offset 815401216 indexer_row_bytes 17
```

Ratio-128 layer without indexer KV:

```text
./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 --slots 32 --kv-dtype f8 --scratch-mib 1536 \
  --dense-kv-slice --layer 3 --slot 7 --position 8192 --indexer off
```

Result:

```text
tp_dense_kv_slice ctx=262144 slots=32 hidden=4096 layer=3 ratio=128 slot=7 position=8192 attn_row=192 indexer_row=18446744073709551615 max_abs=0.000000000
gpu 0 attn_offset 816523456 attn_row_bytes 65 indexer_offset 18446744073709551615 indexer_row_bytes 0
...
gpu 7 attn_offset 816523456 attn_row_bytes 65 indexer_offset 18446744073709551615 indexer_row_bytes 0
```

Logs:

- `logs/from-cluster/sprint230-tp-dense-kv-slice/default-32slot-256k.log`
- `logs/from-cluster/sprint230-tp-dense-kv-slice/layer2-ratio4-indexer.log`
- `logs/from-cluster/sprint230-tp-dense-kv-slice/layer3-ratio128.log`

## Decision

Sprint 230 passes. The separate TP runtime now has concrete resident sharded
KV row ownership for DS4-style ratio-4/indexer and ratio-128 layers at the
target `32` slot / `256K` shape. The row-aligned TP KV layout increases the
runtime KV arena from Sprint 229's coarse estimate to `3707940864` bytes per
GPU, still leaving the TP/EP memory plan viable. This is not serving yet; the
next implementation step is Sprint 231, a bounded EP routed-expert slice that
uses the real low-bit expert kernels inside the separate TP runtime model.
