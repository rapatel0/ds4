# Sprint 378: Compact MoE Decode Gate

## Overview

Implement the next Vision gate after Sprint 377:
`--compact-moe-decode-gate`.

Sprint 377 rejected the narrow batched-paged-attention load target because the
row-family planner showed pending typed-history reloads are already `0` in the
observed compressed/indexer samples. The next measured path is MoE/EP decode
fragmentation.

The concrete first target is not another synthetic routed-kernel sidecar. The
current TP/EP harness already has `--compact-route-compose`, but it cannot run
with real `--model-router-routes`: the route uploader rejects slots whose top-k
experts include more than one expert on the same rank, because compact compose
only tracks one route index per source-rank/slot. That forces real model-router
serving back to the larger all-destination contribution path.

Sprint 378 should make compact model-router compose operational behind a new
default-off gate, then A/B it against the current model-router path.

## Scope

- Add a default-off CLI gate:

```text
--compact-moe-decode-gate
```

- Add launcher/profile plumbing:

```text
DS4_V100_TP_EP_COMPACT_MOE_DECODE=1
tools/ds4-v100-tp-ep-profile.py --compact-moe-decode
```

- Allow model-router routes plus compact compose only when this gate is enabled.
- Replace the one-route-per-source-rank compact compose index with a bounded
  per-source-rank, per-slot route list.
- Keep the current default path unchanged.
- Validate on the V100 pod with same-binary A/B.

## Out Of Scope

- No PP/layer-split work.
- No generic PP/TP scheduler abstraction.
- No MTP in this sprint.
- No broad dtype conversion.
- No CUDA graph/P2P transport rewrite.
- No TP-sharded expert topology rewrite.

## Architecture

Current compact compose shape:

```text
route_index_by_slot[src_rank][slot] -> one route index
```

This is valid for synthetic route layouts where each slot has at most one
route per source rank. It is not valid for true DS4 model-router top-k routes.

Sprint 378 target shape:

```text
route_indices_by_slot[src_rank][slot][k]
route_count_by_slot[src_rank][slot]
```

The compact compose kernel then sums all route rows for each source rank and
slot before adding the local residual, attention, and shared-FFN outputs.

This keeps data movement compact:

```text
copy bytes ~= sum(src_rank.routes * hidden_shard)
```

instead of the all-destination path:

```text
copy bytes ~= gpus * slots * hidden_shard
```

The gate does not change expert math or routing. It only changes compact EP
return composition for duplicate same-rank top-k routes.

## Implementation Plan

### Phase 1: Gate Plumbing

- Add `Options::compact_moe_decode_gate`.
- Parse `--compact-moe-decode-gate`.
- Add launcher env validation and command emission.
- Add profile/matrix flag propagation.
- Print the gate in scaffold/status metadata where existing gate fields are
  reported.

### Phase 2: Multi-Route Compact Compose Metadata

- Add per-rank device buffers:
  - `d_route_indices_by_slot[src_rank]`, sized `slots * top_k`
  - `d_route_count_by_slot[src_rank]`, sized `slots`
- Populate them for synthetic route plans and model-router route plans.
- Remove the model-router duplicate rejection only under
  `--compact-moe-decode-gate`.
- Emit route-shape audit lines:
  - routes per rank
  - active experts per rank
  - max routes per expert
  - duplicate same-rank routes per layer
  - compact return bytes versus all-destination return bytes

### Phase 3: Multi-Route Compact Compose Kernel

- Add a compact compose kernel that reads all route indices for each
  source-rank/slot and accumulates the weighted route-down shards.
- Use it only when:
  - `--compact-moe-decode-gate`
  - `--compact-route-compose`
  - `--model-router-routes`
  - FP32 EP return path
- Preserve the existing compact kernel for synthetic no-duplicate route plans.

### Phase 4: V100 Validation

Build on the V100 pod:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run direct V100 A/B:

```text
control:   --model-router-routes, compact compose off
candidate: --model-router-routes --compact-route-compose --compact-moe-decode-gate
```

Target shape:

```text
32 configured slots
256K context
position 262080
8-32 generated tokens
```

Then run HTTP serving A/B at:

```text
32 active chat requests
32 configured slots
256K context
position 262080
32 generated tokens/request
GPU sampling enabled
```

## Definition Of Done

- `--compact-moe-decode-gate` builds and defaults off.
- Launcher/profile plumbing exists and defaults off.
- Model-router + compact compose is rejected without the gate and accepted with
  the gate.
- Duplicate same-rank route cases are represented in device metadata instead
  of rejected.
- Candidate direct V100 run preserves first token and all-layer decode checksum.
- Candidate HTTP A/B reports client tok/s, server decode tok/s, stage timing,
  GPU utilization, and route-shape audit evidence.
- Sprint doc records explicit PROMOTE, KEEP-OPT-IN, or REJECT.
- `TEMP_STATUS_REPORT_378.md`, `docs/sprints/STATUS.md`, and
  `docs/sprints/VISION.md` are updated.
- Changes and V100 artifacts are committed.

## Decision Rule

Promote only if the gate preserves first token/checksum and improves server
decode tok/s or average GPU utilization at the real `32` slot / `256K` serving
shape.

Keep opt-in if correctness holds but performance is flat/noisy.

Reject if it changes first token/checksum, mishandles duplicate routes, or
regresses EP/compose enough to lower serving throughput.

## Outcome

Sprint 378 is complete.

Implemented:

- `--compact-moe-decode-gate`
- `DS4_V100_TP_EP_COMPACT_MOE_DECODE=1`
- `tools/ds4-v100-tp-ep-profile.py --compact-moe-decode`
- bounded `route_indices_by_slot[src_rank][slot][k]`
- bounded `route_count_by_slot[src_rank][slot]`
- multi-route compact compose kernel
- model-router + compact route validation through the new gate
- route-shape audit output with duplicate slots and compact/all-destination
  byte estimates

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS on gpu-01.

Two implementation bugs were fixed during execution:

- shared-rank cleanup double-freed `d_route_index_by_slot`;
- compact model-router return needed to skip zero-byte peer copies and size
  remote FP32 return buffers for `slots * top_k` route rows.

Direct V100 A/B at `32` slots / `256K` / `position=262080` /
`1` generated token:

| Mode | Return | First token | Decode tok/s | Compose ms | EP ms | Checksum |
|---|---:|---:|---:|---:|---:|---:|
| control: model-router, compact compose off | 0 | 54639 | 62.617354 | 26.025358 | 20.965788 | 6840320333 |
| candidate: compact-MoE compose | 0 | 54639 | 66.481242 | 20.993585 | 17.952991 | 6840320333 |

HTTP serving A/B at `32` requests / `32` slots / `256K` /
`position=262080` / `32` generated tokens/request:

| Mode | HTTP 200 | Response token stream | Client tok/s | Server decode tok/s | Avg GPU util | Compose ms | EP ms |
|---|---:|---|---:|---:|---:|---:|---:|
| control: compact compose off | 32/32 | matched | 37.394075 | 80.812914 | 8.385417% | 19.167728 | 11.236307 |
| candidate: compact-MoE compose | 32/32 | matched | 39.034685 | 81.313535 | 8.559783% | 14.703119 | 11.221664 |

Candidate route-shape evidence:

```text
duplicate_slots=64
max_same_rank_routes=2
all_dest_bytes=4194304
compact_bytes=3145728
```

Decision: PROMOTE for the real model-router compact-compose path.

The global default remains unchanged because model-router routing itself is
still explicitly selected. When model-router routes are enabled, this gate is
the production-compatible compact EP return path and should be used for the
next serving/performance work.

## Risks

- Model-router route upload still uses host staging; compact compose may reduce
  copy/compose while leaving router upload as a remaining bottleneck.
- The top-k route order can include duplicate same-rank routes; accumulation
  order must be deterministic enough to preserve the existing checksum.
- Compact copy size is per source-rank route count; badly imbalanced routing
  may reduce copy bytes less than expected.
- If the real model-router path is not the current serving default, this sprint
  is a production-readiness step more than an immediate default-topline win.
