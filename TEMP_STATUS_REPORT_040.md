# TEMP_STATUS_REPORT_040

Date: 2026-05-24

## Topline

Sprint 328 proved that the TP/EP production KV/cache memory shape from Sprint
327 can be realized as actual CUDA allocations on the 8x V100 32GB node.

This is not serving correctness yet. It is the missing residency proof before
replacing diagnostic f32 compressed-row buffers with a typed TP-sharded
production KV arena.

## What Changed

Added:

- `tools/ds4-v100-tp-kv-arena-smoke.cu`
- `tools/ds4-v100-tp-kv-arena-smoke` Makefile target

The tool allocates, and by default touches, the planner-matching per-GPU arenas:

- resident weight shard
- persistent typed KV shard
- compression-state envelope
- scratch
- collective workspace
- embedding/output global shards

It rejects PP/layer-split options and is TP/EP-only.

## V100 Validation

Build:

- Command: `make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-kv-arena-smoke`
- Result: PASS

Real pack:

- `/workspace/packs/ds4-appliance-full-tm-gated-s181`
- total weight bytes from pack: `156142896212` bytes, about `145.42 GiB`

Allocation runs with page touching enabled:

| Case | Planned alloc / GPU | Free before / GPU | Free after / GPU | Reserve required | Verdict |
|---|---:|---:|---:|---:|---|
| `slots=32`, `ctx=262144`, `kv=f8` | `25.001 GiB` | `31.428 GiB` | `6.424 GiB` | `2.000 GiB` | PASS |
| `slots=1`, `ctx=1048576`, `kv=f8` | `20.558 GiB` | `31.428 GiB` | `10.866 GiB` | `2.000 GiB` | PASS |

After both runs, `nvidia-smi` reported `0 MiB` used on every GPU.

## Interpretation

The target `32` slot / `256K` serving memory budget is physically allocatable on
the V100 node with the real pack weight footprint, F8 typed KV, and a 2 GiB
post-allocation reserve.

This confirms the next implementation should move from diagnostic f32 KV
helpers to a production typed TP-sharded KV arena. The allocator proof removes
VRAM fit as the immediate blocker for 32-slot / 256K TP/EP serving.

## Current Gap

The production arena is proven allocatable but is not yet wired into the
TP/EP layer runtime. The runtime still needs:

- typed arena descriptors for raw SWA, compressed attention rows, and ratio-4
  indexer rows
- row write/read helpers that preserve the Sprint 326 compact-reference diff
  gates
- all-layer validation from production arena reads
- HTTP parity after the runtime reads production KV rather than bounded
  diagnostic f32 buffers

## Artifacts

- `logs/from-cluster/sprint328-tp-kv-arena/cluster/arena-slots32-ctx262144-f8.log`
- `logs/from-cluster/sprint328-tp-kv-arena/cluster/arena-slots1-ctx1048576-f8.log`
