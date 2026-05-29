# Sprint 536 - SPIKE B Preflight, Spill, and Capture Eligibility

Date: 2026-05-29

## Goal

Make the next graph/fusion work measurable after the A4, D1, compact-compose,
and C5 event-handoff cleanup sequence.

## Scope

1. Build the promoted TP/EP appliance with `-Xptxas -v` and record
   register/smem/spill output.
2. Run the current promoted selected-token profile at `32` requests /
   `32` slots / `256K` / `2` tokens.
3. Audit remaining hot-path synchronization and capture blockers.
4. Record the current promoted artifact path for the following SPIKE B
   graph/fusion sprints.

No promotion, feature flag, smoke, or implementation change belongs in this
sprint.

## Artifacts

- Remote workspace: `/workspace/s536-preflight`
- ptxas build log: `/workspace/s536-preflight-ptxas-build.log`
- Promoted-shape profile:
  `/workspace/s536-preflight-profile-r3/none-s536-preflight-selected32-r3`
- Profile summary:
  `/workspace/s536-preflight-profile-r3/none-s536-preflight-selected32-r3/summary.json`
- Nsight Compute attempts:
  `/workspace/s536-preflight-ncu/ncu-window-basic-s536-preflight-ncu-window`
  and
  `/workspace/s536-preflight-ncu-basic/ncu-basic-s536-preflight-ncu-basic`

## Results

### Build And ptxas

Remote build passed:

```text
CUDA_ARCH=sm_70 NVCCFLAGS="-O3 --use_fast_math -arch=sm_70 -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v" make -B -j80 appliance/ds4-v100-tp-ep-appliance
```

Parsed ptxas output:

- Kernel entries: `118`
- Nonzero spill entries: `1`
- Spill site:
  `compressor_pool_emit_slots_kernel`, `255` registers,
  `40` byte stack frame, `40` byte spill stores, `40` byte spill loads.
- Relevant promoted rank-local / attention / compose kernels did not show
  spills in ptxas:
  `hc_apply_reduced_mix_split_kernel` used `40` registers,
  `attention_raw_compressed_window_kernel` used `32` registers and
  `1600` bytes smem,
  `attention_raw_swa_window_kernel` used `32` registers and `1536` bytes smem,
  `pack_rank_major_norm_current_to_routes_scaled_kernel` used `24` registers
  and `1096` bytes smem,
  `compose_next_hidden_compact8_multi_kernel` used `38` registers.

### Promoted-Shape Profile

The low-overhead selected-token profile passed at `32` requests / `32` slots /
`256K` / `2` tokens:

- `http_200=32`
- `output_head_first_token=128819`
- `output_head_finite_bad=0`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- `vram_failures=0`
- `vram_min_free_mib=3852`
- `gpu_steady_util_avg=10.3125`
- `scaffold_ms_per_token=826.829279`

Current domain table from the profile:

| Domain | ms | pct |
|---|---:|---:|
| EP | `532.084435` | `64.35` |
| HC-current input | `244.006877` | `29.51` |
| Compose | `27.754906` | `3.36` |
| Final HC | `22.957099` | `2.78` |

Top fine-grained buckets:

| Bucket | ms | pct |
|---|---:|---:|
| pre-EP attention state | `54.474073` | `6.59` |
| pre-EP HC-current | `54.284037` | `6.57` |
| pre-EP post-attention FFN input | `36.669077` | `4.43` |
| pre-EP attention projection | `34.578090` | `4.18` |
| pre-EP attention output | `33.495894` | `4.05` |
| HC-current FFN router | `29.964544` | `3.62` |
| pre-EP raw read | `29.283190` | `3.54` |

### Nsight Compute

`ncu` is installed in the V100 container:

```text
NVIDIA Nsight Compute 2023.2.2.0
```

Two short filtered attempts were made:

1. `ncu-window-basic`, `32` selected-token requests, `40` matching launches:
   completed functionally but reported `No kernels were profiled`.
2. `ncu-basic`, `8` selected-token requests, `20` matching launches:
   failed during request handling after Nsight reported
   `Profiling failed because a driver resource was unavailable`.

Decision: ptxas spill data is usable. Nsight Compute occupancy collection is a
tooling blocker on the current node state and should be retried during the
tuning sprint with DCGM/profiling-resource contention cleared.

### Capture Blockers

Static hot-code audit:

- `cudaDeviceSynchronize`: `31` active `engine/*.cu` hits.
- `cudaStreamSynchronize`: `72` active `engine/*.cu` hits, including
  diagnostic helpers and remaining decode-loop / EP compose / typed-history
  boundaries.
- `cudaEventSynchronize`: `9` active `engine/*.cu` hits.
- `ds4_peer_copy_async`: `0`.
- `ncclSend` / `ncclRecv`: `0`.
- `cudaMemcpyPeerAsync`: `1`, in `engine/context_cuda.cu` relay code outside
  the promoted TP/EP serving hot path.
- `enqueue_graph_f32_copy_between_devices`: `4` hits including helper
  definition and remaining diagnostic/copy-wrapper call sites.
- `enqueue_graph_f32_copy_from_device0`: `11` hits including helper
  definition and remaining broadcast/copy-wrapper call sites.

The promoted selected-token profile confirms the runtime invariant that matters
for C1: peer-copy accounting remains zero and NCCL graph SYS edges remain zero.

## Decision

Sprint 536 closes as a measurement/control sprint. No code promotion is
expected.

The reusable control artifact for the next SPIKE B sprint is:

```text
/workspace/s536-preflight-profile-r3/none-s536-preflight-selected32-r3/summary.json
```

C1 can start next per the existing roadmap, with these known constraints:

- Preserve the no-SYS NCCL topology and peer-copy-zero invariant.
- Do not count a graph speedup until selected-token / generated-sequence parity
  passes against the Sprint 536 control.
- Treat remaining host syncs as C1/C2 ordering targets, not as license to add
  broad device synchronizes.
- Retry `ncu` occupancy during tuning after profiling-resource contention is
  cleared.
