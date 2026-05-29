# Sprint 526 - A4 Finish Rank-Major Consumers

Date: 2026-05-28

## Goal

Finish SPIKE B A4 by making the remaining post-attention FFN consumers use the
rank-major post-attention current layout, then remove the carried slot-major
FFN-norm/full-current staging that only existed for those consumers.

This sprint is the required cleanup before C1 graph capture work. It should
turn the existing partial rank-major conversion into a deleted staging path,
not add another long-lived diagnostic flag.

## Context

- `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` now define the active
  order as A4 first, then output-head A1, sync-point reduction, compact EP
  compose NCCL, then C1.
- A1-A3 are already promoted:
  - A2 HC mix row-parallel all-reduce from Sprint 478.
  - A3 router all-reduce from Sprint 480.
  - Attention-projection rank-major input from Sprint 483, although that older
    sprint called it "A6 PATH 4"; structurally it is A4.
- `engine/post_attention_ffn.cu` already contains the remaining rank-major
  machinery:
  - `d_post_attn_full_rank_major`
  - `fill_two_hidden_inputs_half_from_rank_major_norm_kernel`
  - `pack_rank_major_norm_current_to_routes_kernel`
  - `pack_rank_major_norm_current_to_routes_scaled_kernel`
- The remaining carry-forward path is the slot-major FFN norm/full-current
  gather/broadcast used when rank-major shared/route consumers are not fully
  active or when parity/diagnostic gates force the legacy path.

## Scope

1. Make the post-attention FFN shared input consumer rank-major by default.
2. Make the post-attention routed FFN input consumer rank-major by default.
3. Remove the promoted-path dependency on slot-major `hc->d_ffn_normed` and
   `r.d_current_full` in `engine/post_attention_ffn.cu`.
4. Remove or quarantine dead full-current gather / slot-major transpose code
   only after the promoted path no longer needs it.
5. Keep diagnostic/parity-only code explicit if it is still needed for future
   debugging, but do not leave the promoted path gated behind confusing flag
   combinations.

## Non-Goals

- Do not touch MTP.
- Do not start C1 graph capture in this sprint.
- Do not implement output-head A1 or compact EP compose NCCL here.
- Do not rerun unrelated controls unless the current binary/defaults invalidate
  prior promoted control evidence.

## Implementation Plan

1. Inspect `engine/post_attention_ffn.cu` and related defaults to identify the
   exact conditions that keep `needs_slot_major_ffn_norm` and
   `post_ffn_slot_major_broadcast` alive in the promoted path.
2. Promote the rank-major shared/route FFN input path in runtime defaults and
   launcher/profile defaults as needed.
3. Simplify `engine/post_attention_ffn.cu` so the default promoted path:
   - all-gathers `d_post_attn_shard` into `d_post_attn_full_rank_major`,
   - fills shared gate/up inputs from rank-major normalized data,
   - packs routed input from rank-major normalized data,
   - does not gather post-attention current into slot-major `hc->d_current_full`,
   - does not broadcast slot-major `hc->d_ffn_normed` back to ranks.
4. Preserve parity diagnostics only behind explicit diagnostic gates.
5. Build locally, then sync the changed files to the V100 node and build the
   TP/EP appliance there.
6. Run the target selected-token/tolerance gate against the latest promoted
   control artifact unless a control refresh is required by the changed
   defaults.

## Validation

Required build checks:

- `git diff --check`
- Local representative build for touched engine/appliance targets.
- Remote V100 build:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Required correctness/performance gate:

- Same-binary target-shape selected-token/tolerance or HTTP A/B against the
  latest promoted control leg, reusing prior promoted control artifacts unless
  invalidated.
- Agreement must satisfy the current relaxed arithmetic policy.
- `peer_copy_ops=0` and `peer_copy_sys_bytes=0` must remain true on the
  promoted path.
- Capture/report whether slot-major FFN norm and post-FFN slot-major broadcast
  are absent from the promoted path.
- Record server decode tok/s, request-window utilization, HC-current/post-
  attention staging timers, and launch/sync deltas where available.

## Definition of Done

- The promoted TP/EP path uses rank-major post-attention FFN shared and route
  inputs by default.
- The promoted path no longer needs post-attention slot-major FFN norm or
  slot-major FFN-norm broadcast.
- Any remaining legacy slot-major path is diagnostic-only with an explicit
  reason and not part of the default appliance path.
- Local and remote builds pass.
- V100 correctness gate passes or the sprint records a concrete blocker with
  enough evidence to continue.
- `docs/sprints/SPRINT-526.md` is updated with final changes and validation.

## Changes

- Promoted `routed_ffn_rank_major_input_gate=true` in
  `engine/runtime_options.cuh`.
- Removed `post_attention_skip_slot_major_ffn_norm_gate` from active runtime
  options, scaffold logging, and profile summaries. Sprint 456 remains in the
  historical record as the rejected narrow skip experiment.
- Moved `hc->d_current_full` and `hc->d_ffn_normed` preconditions behind the
  legacy/diagnostic slot-major path in `engine/post_attention_ffn.cu`, so the
  promoted rank-major path no longer requires those slot-major buffers.
- Kept the explicit diagnostic `post_attention_slot_major_ffn_norm_gate` path
  available for parity/debugging.
- Added `--max-requests N` parsing to the extracted appliance option parser so
  the current profile launcher and appliance binary agree after the structural
  extraction.

## Validation Results

Local:

- `git diff --check`: pass.
- Active-code search for `post_attention_skip_slot_major_ffn_norm` returns only
  historical sprint/archive/vision references, no live code or tooling.

Remote V100:

- Synced repo to
  `/localpool/ds4/workspace/s526-a4-rank-major` with `rsync`, excluding
  `.git/`, `research/`, build artifacts, and object files.
- Build passed inside the CUDA 12.2 container:
  `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`.
- Final selected-token gate:
  `/localpool/ds4/workspace/s526-a4-rank-major-selected32-final`
  - `http_200=32`
  - `output_head_first_token=128819`
  - `client_generated_tok_s=9.12082497906123`
  - `decode_domain_total_ms=1934.426273`
  - `scaffold_projected_slot_step_tok_s=16.542373`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`
  - `vram_failures=0`
  - `vram_min_free_mib=3826`
  - `scaffold_routed_ffn_rank_major_input_gate=1`
  - `scaffold_post_attention_slot_major_ffn_norm_gate=0`

Server log spot checks from the final artifact show the promoted path on every
checked layer:

```text
rank_major_input 1
rank_major_shared_input 1
rank_major_route_input 1
slot_major_ffn_norm 0
```

## Decision

Sprint 526 completes SPIKE B A4 for the served TP/EP path. The post-attention
FFN shared and routed consumers are rank-major by default, and the promoted path
no longer depends on slot-major FFN norm staging or the rejected skip flag. The
next SPIKE B sprint is D1/output-head A1 pattern: apply the A2 rank-local NCCL
template at the model boundary.

This sprint also reaffirms the cleanup rule: temporary smokes and flags are
evaluation scaffolding, not permanent product surface. Promotion moves the
behavior into the main path and removes the accumulated gate unless the sprint
records a concrete diagnostic-only reason to keep it.
