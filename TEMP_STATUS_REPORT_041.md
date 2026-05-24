# TEMP_STATUS_REPORT_041

Date: 2026-05-24

## Topline

Sprint 329 aligned the TP/EP planner and allocation smoke with the physical
row-sharded KV layout that `ds4_v100_tp_runtime` already uses.

The target still fits. The corrected physical KV budget is about
`58.46 MiB/GPU` larger than the ideal aggregate-sharded lower bound.

## What Changed

Updated:

- `tools/ds4-v100-plan-tp.c`
- `tools/ds4-v100-tp-kv-arena-smoke.cu`

The resident budget now uses physical row-sharded KV bytes:

```text
physical row-sharded KV / GPU = 3707940864 bytes
ideal aggregate-sharded KV / GPU = 3646642176 bytes
overhead / GPU = 61298688 bytes
```

The planner still prints the ideal value, but only as a comparison. Fit and
admission now use the physical value.

## V100 Validation

Build:

- Command: `make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-plan-tp tools/ds4-v100-tp-kv-arena-smoke tools/ds4-v100-tp-runtime-smoke`
- Result: PASS

Target `32` slots / `256K` / F8, real pack:

| Source | KV bytes / GPU | Result |
|---|---:|---|
| planner physical row-sharded | `3707940864` | PASS |
| runtime smoke | `3707940864` | PASS |
| arena smoke | `3707940864` | PASS |

Arena allocation result:

| Planned alloc / GPU | Free before / GPU | Free after / GPU | Reserve required | Verdict |
|---:|---:|---:|---:|---|
| `25.058 GiB` | `31.428 GiB` | `6.366 GiB` | `2.000 GiB` | PASS |

Admission tiers after correction:

| Context | Max slots | Per-GPU total at max |
|---:|---:|---:|
| `131072` | `125` | `31.97 GiB` |
| `262144` | `62` | `31.87 GiB` |
| `524288` | `31` | `31.86 GiB` |
| `1048576` | `15` | `31.53 GiB` |

## Interpretation

The previous Sprint 327/328 planner numbers were slightly optimistic because
they sharded aggregate KV bytes. The actual runtime shards each row across TP
ranks, so per-row quantization/block padding must be included before summing.

This correction is small but important. The production serving target remains
valid:

- `32` slots / `256K` fits with `27.058 GiB/GPU` planned budget
- no-reserve allocation is `25.058 GiB/GPU`
- allocation leaves `6.366 GiB/GPU` free on the current pod

## Current Gap

Memory accounting now matches the runtime allocator. The next gap is execution
integration:

- full-layer TP/EP attention still uses f32 diagnostic raw/compressed row
  buffers
- production typed row store/load kernels need to feed those attention reads
- compact-reference diff gates should stay active while the backing storage
  changes

## Artifacts

- `logs/from-cluster/sprint329-row-sharded-kv/cluster/plan-slots32-ctx262144-f8.md`
- `logs/from-cluster/sprint329-row-sharded-kv/cluster/plan-slots32-ctx262144-f8.json`
- `logs/from-cluster/sprint329-row-sharded-kv/cluster/runtime-slots32-ctx262144-f8.log`
- `logs/from-cluster/sprint329-row-sharded-kv/cluster/arena-slots32-ctx262144-f8.log`
