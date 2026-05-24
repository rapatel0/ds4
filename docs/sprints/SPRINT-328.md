---
sprint: 328
title: TP/EP Production KV Arena Allocation Smoke
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 328 - TP/EP Production KV Arena Allocation Smoke

## Goal

Prove that the Sprint 327 TP/EP compressed-KV memory contract can be realized
as actual CUDA allocations on the 8x V100 32GB node.

## Why This Sprint

Sprint 327 made the planner state the target memory shape numerically:
`32` slots, `256K` context, F8 typed KV, TP8 sharding, and full weight
residency fit with about `5 GiB` headroom after reserve. That is necessary but
not enough. The runtime needs allocator evidence from the real GPUs before we
replace diagnostic f32 buffers with production typed arenas.

This sprint is the bridge from "planner says it fits" to "CUDA can reserve the
target shape on every V100."

## Scope

TP/EP only. No PP/layer-split variants. No MTP. No serving semantics change in
this sprint.

## Implementation Plan

1. Add a standalone CUDA tool:

   - `tools/ds4-v100-tp-kv-arena-smoke.cu`

2. The tool should mirror the planner's per-GPU budget:

   - TP-sharded weight footprint
   - TP-sharded persistent KV
   - TP-sharded compression-state envelope
   - scratch
   - collectives
   - global embedding/output-head shards
   - reserve/headroom check

3. The tool should allocate those arenas on all visible GPUs and optionally
   touch pages with a tiny CUDA kernel.

4. Validate on the V100 pod:

   - `32` slots / `256K` / F8 / real pack weight total
   - `1` slot / `1M` / F8 / real pack weight total

5. Record cluster logs locally and update the vision/status docs.

## Definition of Done

- [x] CUDA build target exists for `tools/ds4-v100-tp-kv-arena-smoke`.
- [x] The tool prints planner-matching component byte budgets.
- [x] The tool allocates the target arenas on 8 GPUs.
- [x] The tool verifies remaining free memory is at least the configured
      reserve.
- [x] The `32` slot / `256K` / F8 target passes on the V100 pod.
- [x] The `1` slot / `1M` / F8 target passes on the V100 pod.
- [x] Cluster logs are copied under `logs/from-cluster/`.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_040.md`, and this sprint doc are
      updated and committed.

## Outcome

Added `tools/ds4-v100-tp-kv-arena-smoke.cu`, a CUDA allocation smoke that
mirrors the Sprint 327 TP/EP planner budget and allocates all major per-GPU
arenas:

- dummy resident weight shard
- typed persistent KV shard
- compression-state envelope
- scratch
- collective workspace
- global embedding/output shards

The tool optionally touches the allocated pages with a CUDA kernel. The V100
validation used touching enabled.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-kv-arena-smoke
```

Result: PASS.

Real-pack allocation runs:

| Case | Planned alloc / GPU | Free after / GPU | Reserve required | Verdict |
|---|---:|---:|---:|---|
| `slots=32`, `ctx=262144`, `kv=f8` | `25.001 GiB` | `6.424 GiB` | `2.000 GiB` | PASS |
| `slots=1`, `ctx=1048576`, `kv=f8` | `20.558 GiB` | `10.866 GiB` | `2.000 GiB` | PASS |

Both runs allocated and touched the arenas on all eight GPUs. The node returned
to `0 MiB` used on every GPU after the runs.

Artifacts:

- `logs/from-cluster/sprint328-tp-kv-arena/cluster/arena-slots32-ctx262144-f8.log`
- `logs/from-cluster/sprint328-tp-kv-arena/cluster/arena-slots1-ctx1048576-f8.log`

## Next Step

Wire the production typed KV arena into the TP/EP runtime:

- replace bounded diagnostic f32 compressed-row buffers with typed arena slices
- keep layer-local row descriptors and active aliases
- preserve the Sprint 326 compact-reference diff gate while reading from the
  production arena
- rerun all-layer compressed-history gates at `32` slots / `256K`

## Risks

- Other processes on the GPU can make an otherwise valid allocation fail. If
  that happens, the evidence should include `nvidia-smi` and free-memory
  snapshots instead of hiding the failure.
- `cudaMalloc` allocation success proves residency budget shape, not serving
  correctness. The next sprint still needs to wire production KV arenas into
  the TP/EP runtime.
- The dummy weight allocation is intentionally conservative. Real weight
  binding may fragment differently, so the serving allocator should keep the
  same reserve discipline.
