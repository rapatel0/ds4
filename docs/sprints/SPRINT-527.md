# Sprint 527 - D1 Output-Head A1 Pattern

Date: 2026-05-28

## Goal

Apply the SPIKE B D1 model-boundary cleanup: replace the output-head
gather-to-GPU0 / centralized RMS+head-mix / broadcast sequence with the same
rank-local partial plus NCCL collective pattern that A2 promoted inside the
layer stack.

This is the next sprint after A4. It should land in the main output-head path,
not as another permanent feature flag.

## Context

- Sprint 526 completed A4 for the served TP/EP path. Post-attention FFN shared
  and routed consumers are rank-major by default, and slot-major FFN norm is no
  longer on the promoted path.
- `SPIKE_B_STEERING.md` now lists D1 output-head A1 pattern as the next
  bankable NCCL cleanup before sync-point reduction, compact EP compose, and
  C1 graph capture.
- `engine/output_head.cu` still does model-boundary work centrally on GPU0:
  gather final HC shards to GPU0, RMS norm over `[slots,4,4096]`, four-row head
  mix, output HC weights, weighted HC sum, output RMS norm, and then broadcasts
  the full normalized embedding back to all ranks for vocab-sharded projection.

## Scope

1. Split output-head control weights by hidden shard during
   `open_shared_output_head`:
   - rank-local `hc_head_fn` shard for the four head-mix outputs,
   - rank-local `output_norm.weight` shard.
2. Replace GPU0 final-HC gather and centralized head prep with:
   - per-rank local max and partial head-mix over `d_final_hc_shard`,
   - NCCL all-reduce max and partial mix,
   - per-rank stable sumsq over the same final-HC shard,
   - NCCL all-reduce sumsq,
   - per-rank output HC weight calculation and weighted embedding shard.
3. Replace GPU0 final output RMS with:
   - per-rank embedding-shard sumsq,
   - NCCL all-reduce sumsq,
   - per-rank output-norm shard application.
4. Replace the GPU0 full-embedding broadcast with an NCCL all-gather from
   rank-local normalized embedding shards into each rank's projection input.
5. Keep projection/top-1 behavior unchanged for this sprint.

## Non-Goals

- Do not touch MTP.
- Do not start C5 host-sync reduction outside the output-head sequence.
- Do not change output vocab projection sharding.
- Do not add a long-lived runtime flag. One-off smoke/evaluation scaffolding is
  allowed only if removed before promotion.

## Implementation Plan

1. Extend `SharedOutputHead` with rank-local scratch/control buffers.
2. Add small output-head kernels for local HC max/mix partials, local stable
   HC sumsq, reduced head-weight application, local weighted embedding shard,
   output embedding shard sumsq, and output RMS+weight shard.
3. In `run_shared_output_head_from_rank_hc`, replace the GPU0 gather/prep and
   rank0 broadcast section with rank-local kernels plus NCCL all-reduce/all-
   gather on each rank's output-head stream.
4. Preserve existing output-head timing fields, mapping the old gather bucket to
   the rank-local HC collectives and the broadcast bucket to the final NCCL
   all-gather.
5. Validate against the latest promoted control leg at the selected-token shape.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new D1 feature flag or permanent smoke gate.

Required remote checks:

- Sync changed tree to the V100 host via `rsync`, excluding `.git/`,
  `research/`, build artifacts, and object files.
- Remote CUDA build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Run a small selected-token smoke if needed to catch parser/runtime failures.
- Run the target selected-token gate against the promoted reference shape:
  `32` requests / `32` slots / `256K` / `2` tokens.

Gate requirements:

- `http_200=32`
- selected first token remains compatible with the promoted control artifact
  for the same shape.
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- `output_head_finite_bad=0`
- Record output-head total/prep/broadcast/projection/top1 timings and sync
  counts.

## Definition of Done

- The served output-head path no longer gathers final HC to GPU0 for RMS/mix.
- The served output-head path no longer centralizes output embedding RMS on
  GPU0.
- The remaining movement of normalized embedding into projection inputs is an
  NCCL all-gather, not a GPU0 broadcast.
- No permanent flag or one-off smoke scaffold is left behind.
- Local and remote builds pass.
- V100 selected-token gate passes or this sprint records a concrete blocker and
  leaves the promoted path unchanged.

## Changes

- Split `hc_head_fn`, `hc_head_base`, `hc_head_scale`, and
  `output_norm.weight` into rank-local device buffers during
  `open_shared_output_head`.
- Replaced the GPU0 final-HC gather and centralized HC RMS/head-mix with
  per-rank local max/mix partials, NCCL max/sum all-reduces, and per-rank stable
  sumsq/weight application.
- Replaced the GPU0 output embedding RMS with per-rank weighted embedding
  shards, NCCL max/sum all-reduces, and rank-local output-norm shards.
- Replaced the GPU0 full-embedding broadcast with NCCL all-gather into
  rank-major embedding buffers followed by rank-major-to-slot-major conversion
  for the existing vocab-sharded projection.
- Kept vocab projection and top-1 selection unchanged.
- Did not add a runtime flag or one-off smoke scaffold.

## Validation Results

Local:

- `git diff --check`: pass.
- Active-code search: no new D1/output-head feature gate or permanent smoke
  scaffold.

Remote V100:

- Synced repo to `/localpool/ds4/workspace/s527-output-head-a1`.
- Build passed inside the CUDA 12.2 container:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`.
- Correctness guardrail: a simplified non-stable RMS shortcut was tested and
  rejected because it changed the selected token from `128819` to `68338`; that
  shortcut is not in the committed path.
- Final selected-token gate:
  `/localpool/ds4/workspace/s527-output-head-a1-selected32-final`
  - `http_200=32`
  - `output_head_first_token=128819`
  - `output_head_finite_bad=0`
  - `client_generated_tok_s=8.949370821310039`
  - `decode_domain_total_ms=1699.656381`
  - `scaffold_projected_slot_step_tok_s=18.827335`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `vram_failures=0`
  - `vram_min_free_mib=3830`

Output-head comparison against the Sprint 526 promoted control artifact
`/localpool/ds4/workspace/s526-a4-rank-major-selected32-final`:

| Metric | Sprint 526 control | Sprint 527 candidate |
|---|---:|---:|
| first token | `128819` | `128819` |
| output head total | `9.114365 ms` | `10.240521 ms` |
| gather / HC collective bucket | `0.239104 ms` | `0.590978 ms` |
| prep bucket | `0.110605 ms` | `0.927426 ms` |
| broadcast / embedding all-gather bucket | `0.490656 ms` | `0.298473 ms` |
| projection | `7.712662 ms` | `7.915620 ms` |
| top1 | `0.561009 ms` | `0.507597 ms` |
| device sync count | `26` | `16` |

## Decision

Promote as a structural/C1-readiness cleanup, not as a direct output-head
throughput win. The sprint removes GPU0-centralized output-head prep and drops
host-visible output-head synchronization count, but the stable NCCL reduction
sequence is slower than the old centralized prep at this shape. Keep it because
it simplifies the capture surface and satisfies D1's de-centralization goal;
future C5/A5-style fusion should collapse the extra output-head prep kernels
and collectives if this boundary remains visible.

The next SPIKE B sprint is C5 sync-point reduction.
