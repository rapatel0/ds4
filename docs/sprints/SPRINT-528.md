# Sprint 528 - C5 Sync-Point Reduction Pass 1

Date: 2026-05-28

## Goal

Start SPIKE B C5 by removing host-wide synchronization from the served TP/EP
hot path where the dependency is local and can be represented by a stream/event
wait. This first pass is intentionally narrow: reduce output-head and adjacent
decode synchronization without introducing flags or changing math.

## Context

- Sprint 526 completed A4 rank-major post-attention FFN consumers.
- Sprint 527 completed D1 output-head de-centralization, but the output-head
  path still reports `device_sync_count=16` because projection/top-1 validation
  waits use `cudaDeviceSynchronize()`.
- `SPIKE_B_STEERING.md` lists C5 as the next priority before compact EP compose
  and C1 graph capture.
- The goal is structural graph-readiness and host-round-trip cleanup, not a
  per-sprint throughput claim.

## Scope

1. Audit hot-path `cudaDeviceSynchronize()` and `cudaStreamSynchronize()` calls
   in:
   - `engine/output_head.cu`
   - `engine/decode_loop.cu`
   - `engine/attention_projection.cu`
   - `engine/attention_output.cu`
   - `engine/post_attention_ffn.cu`
   - `engine/hc_current.cu`
   - `engine/ep_compose.cu`
2. Classify each sync as:
   - required host observation,
   - diagnostic-only,
   - cross-stream dependency that can become an event,
   - device-wide wait that can become stream-scoped.
3. Implement only the low-risk first pass:
   - replace output-head device-wide waits used for projection timing and top-1
     readiness with stream/event-scoped synchronization,
   - keep host waits only where the CPU must read selected tokens/logits,
   - preserve timing output and selected-token behavior.
4. Do not add runtime flags. If a temporary diagnostic branch is needed during
   evaluation, remove it before promotion.

## Non-Goals

- Do not touch MTP.
- Do not start C1 graph capture.
- Do not rewrite EP compose transport; that is Sprint 529/B2.
- Do not remove diagnostics that intentionally copy tensors to host outside the
  served path.

## Validation

Required local checks:

- `git diff --check`
- active-code search confirms no new C5 feature flag or permanent smoke gate.

Required remote checks:

- Remote CUDA build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Selected-token gate at `32` requests / `32` slots / `256K` / `2` tokens.

Gate requirements:

- `http_200=32`
- selected first token remains compatible with the promoted control artifact
  for the same shape.
- `output_head_finite_bad=0`
- `peer_copy_ops=0`
- `peer_copy_sys_bytes=0`
- `nccl_graph_sys_edge_count=0`
- Output-head `device_sync_count` decreases relative to Sprint 527 or the
  sprint records why a host/device wait is required.

## Definition of Done

- The first C5 sync inventory is recorded in this sprint doc.
- Low-risk output-head device-wide waits are replaced with stream/event-scoped
  waits.
- No new flag or smoke scaffold is left behind.
- Local and remote builds pass.
- V100 selected-token gate passes or the sprint records a concrete blocker and
  leaves the promoted path unchanged.

## Sync Inventory

Hot-path sync sites found by `rg`:

- `engine/output_head.cu`: projection timing and top-1 readiness used
  device-wide waits in the served diagnostic output-head path. This sprint
  addresses those waits.
- `engine/decode_loop.cu`: pass/fence helpers still use stream waits around
  rank, dense, and copy streams. These need a broader event-dependency pass and
  are left for C5 pass 2.
- `engine/attention_projection.cu`, `engine/attention_output.cu`,
  `engine/post_attention_ffn.cu`, `engine/hc_current.cu`,
  `engine/ep_compose.cu`: several waits remain, but many are tied to
  diagnostics, fallback paths, or cross-stream phase boundaries. They require
  per-site dependency review rather than mechanical replacement.
- `engine/compressed_kv_step.cu` has many waits but compressed KV is not active
  in the current served profile (`compressed_kv_layers=0`), so it is not part of
  this first pass.

## Changes

- Replaced output-head projection `cudaDeviceSynchronize()` with
  `cudaEventSynchronize(projection_stop[gpu])`, preserving kernel timing while
  avoiding a device-wide wait.
- Replaced output-head top-1 device-wide waits plus synchronous D2H copies with
  stream-ordered `cudaMemcpyAsync()` into existing pinned host buffers and a
  stream-scoped wait before CPU token/logit reduction.
- Kept output-head math, projection, and top-1 kernels unchanged.
- Did not add a runtime flag or smoke scaffold.

## Validation Results

Local:

- `git diff --check`: pass.
- Active-code search: no new C5 feature flag or permanent smoke gate.

Remote V100:

- Synced repo to `/localpool/ds4/workspace/s528-sync-pass1`.
- Build passed inside the CUDA 12.2 container:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`.
- Final selected-token gate:
  `/localpool/ds4/workspace/s528-sync-pass1-selected32`
  - `http_200=32`
  - output-head server line first token: `128819`
  - `output_head_finite_bad=0`
  - `client_generated_tok_s=9.949731064713747`
  - `decode_domain_total_ms=1717.982567`
  - `scaffold_projected_slot_step_tok_s=18.626499`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `vram_failures=0`
  - `vram_min_free_mib=3830`

Output-head sync counters:

| Metric | Sprint 527 | Sprint 528 |
|---|---:|---:|
| `device_sync_count` | `16` | `0` |
| `stream_sync_count` | `0` | `8` |
| `event_sync_count` | `0` | `8` |

Server output-head line:

```text
tp_ep_diagnostic_output_head ... first_token 128819 ... device_sync_count 0 stream_sync_count 8 event_sync_count 8 ... PASS
```

The profile summary's `output_head_first_token` field is taken from a response
`selected_token` value and can reflect the continuation token for a slot. The
authoritative output-head server line preserved the first generated token
`128819`.

## Decision

Promote C5 pass 1. The output-head path no longer uses device-wide waits for
projection timing or top-1 readback, and the selected-token guardrail remains
clean. C5 is not complete globally; the next sync-point work should tackle the
rank/dense/copy stream phase waits in `engine/decode_loop.cu` and the per-stage
waits in attention/post-attention files.
