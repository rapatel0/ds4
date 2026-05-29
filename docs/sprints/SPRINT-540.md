# Sprint 540 - C1 Warmed Graph Serving Decision

Date: 2026-05-29

## Goal

Decide whether the Sprint 539 route-stable graph suffix is a serving
performance promotion or only a correctness/cache milestone.

## Starting Evidence

- Sprint 539 restored strict selected-token parity for graph suffix replay at
  `8` requests / `8` slots / `256K` for `4` and `8` generated tokens.
- It restored persistent cache hits (`43`) and removed position invalidations
  (`0`) by using graph-only fixed-capacity post-attention route geometry.
- Reduced timing was mixed and startup dominated. It is not valid promotion
  evidence.

## Scope

1. Run a warmed eager selected-token control at a comparable long-generation
   shape.
2. Run the graph suffix candidate at the same shape with:
   `--decode-cudagraph-gate`,
   `--decode-cudagraph-replay-probe-gate`,
   `--decode-cudagraph-persistent-replay-gate`, and
   `--decode-cudagraph-suffix-stage compose_eager_final_hc`.
3. Compare generated token sequences/checksums, request-window throughput,
   cache hits, graph invalidations, and transport invariants.
4. Promote only if correctness holds and request-window performance is
   non-regressing enough to justify the fixed-capacity padding cost.

## Measurement Shape

- Endpoint: `/v100/selected-token`
- Requests: `32`
- Slots: `32`
- Context: `262144`
- Position: `262080`
- Generated tokens: `64`
- Tool: `none`
- Startup warmup: profile default (`auto`)

This isolates initialization by using the profile harness lifecycle/request
window fields and enough generated tokens to make startup and first-token
effects less dominant.

## Non-goals

- No MTP work.
- No new permanent graph flags.
- No default promotion from short probes.
- No stochastic prompt-level performance evidence.

## Execution

Artifacts:

- Eager control:
  `/workspace/s540-warmed-graph-artifacts/none-s540-eager32x64-p262080`
- Graph candidate:
  `/workspace/s540-warmed-graph-artifacts/none-s540-graph32x64-compose-stable-p262080-serverargs-h2180dc1d`

Correctness:

- Both legs served `32/32` HTTP 200.
- First output-head token matched: `107027`.
- All `32` generated token sequences and decode-step checksums matched exactly.
- Graph replay:
  - `graph_audit_persistent_cache_hits=43`
  - `graph_audit_persistent_cache_misses=0`
  - `graph_audit_persistent_invalidate_position=0`
  - `graph_audit_replay_succeeded=43`
  - `graph_audit_replay_attempted=43`
- Transport invariant held:
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

Performance:

| Metric | Eager | Graph | Result |
|---|---:|---:|---:|
| Startup/readiness | `103.175930s` | `103.183448s` | same |
| Request window | `99.446247s` | `90.181067s` | graph `+10.3%` |
| Client generated tok/s | `20.594068731` | `22.709903571` | graph `+10.3%` |
| Scaffold ms/token | `832.498621` | `666.058962` | graph `+20.0%` |
| Scaffold slot-step tok/s | `38.438502` | `48.043795` | graph `+25.0%` |
| Request steady SM util | `18.104592%` | `14.677721%` | lower |
| Min free VRAM | `3852 MiB` | `3726 MiB` | ok |

Decision:

- Promote graph suffix replay as the TP/EP launcher default.
- Keep an explicit operational opt-out:
  `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=0`.
- The promoted launcher adds:
  `--decode-cudagraph-gate`,
  `--decode-cudagraph-replay-probe-gate`,
  `--decode-cudagraph-persistent-replay-gate`, and
  `--decode-cudagraph-suffix-stage compose_eager_final_hc`.
- Continue to treat full graph capture and MTP as separate future work.

Launcher checks:

- `tools/ds4-v100-run-tp-ep-appliance.sh --print-command --allow-missing`
  emits the graph suffix replay args by default.
- `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=0
  tools/ds4-v100-run-tp-ep-appliance.sh --print-command --allow-missing`
  emits no graph suffix replay args.
- `bash -n tools/ds4-v100-run-tp-ep-appliance.sh`: PASS
- `git diff --check`: PASS
