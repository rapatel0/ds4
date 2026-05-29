# Sprint 549 - C1 Rejected Padding Knob Cleanup

Date: 2026-05-29

## Goal

Remove rejected post-attention padding experiments from active code so the
promoted graph-stable route path is the only supported implementation surface.
This sprint is cleanup only; it makes no throughput claim.

## Implementation

Removed the rejected/default-off scaffolding for:

- `post_attention_device_actual_route_sync_gate`
- `post_attention_static_rank_route_cap`
- `post_attention_static_executor_route_cap`
- `post_attention_static_compose_route_cap`
- `post_attention_masked_compact_copy_gate`

The promoted fixed-capacity route plan remains default-on through
`post_attention_fixed_capacity_route_plan_gate`.

Active code now always uses the fixed graph-visible route capacity in the
post-attention graph-order route plan. The old host-synced actual-route update,
static cap audit output, executor/compose row caps, masked compact copy branch,
and the masked-copy CUDA kernel were deleted. The token-major scaffold summary
and profile parser were trimmed so they no longer report retired knobs.

## Validation

Active-code grep:

- No matches for the removed symbols in `engine/`, `kernels/v100/`, `tools/`,
  or `appliance/`.

Remote workspace:

- `/workspace/s549-padding-cleanup`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- PASS

No serving/profile validation was run because this sprint removes inactive
rejected branches and preserves the promoted path. The next performance sprint
should validate the actual fixed-shape/device-state change it introduces.

## Decision

Promote the cleanup. Do not reintroduce static route caps or masked compact
copy as graph-padding levers; prior sprints showed those shapes can preserve
overflow audits while changing tokens or failing to transfer performance.

The next C1 work remains one of:

- Full-shape device-masked routed executor/compose work that preserves
  graph-visible shapes while skipping inactive rows internally.
- A typed-KV/runtime device-state refactor plan for full-capture reuse.
