# Sprint 461: TP/EP Graph Event-Order Router Dependency Fix

## Objective

Fix the first concrete graph event-order dependency hole found after Sprint
460: rank-major model-router logits are consumed on the control stream without
waiting for rank-stream NCCL allgather completion when `--decode-cudagraph` is
enabled.

## Rationale

Sprint 460 showed no-replay graph mode changes output tokens and regresses
HC-current gather. Inspection found a specific eager-vs-graph ordering
difference in `run_model_router_rank_major_logits`:

- eager mode synchronizes rank streams after router-logits NCCL allgather
- graph mode skips that synchronization
- the control stream then runs `router_logits_rank_major_to_slot_major_kernel`
  against rank-0's gathered logits

That is a real missing dependency and can directly corrupt routing decisions.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add the missing graph-mode control-stream wait after rank-major router
  allgather.
- Rebuild on V100.
- Re-run the same 8-slot / 256K / 3-token no-replay graph A/B.

## Definition of Done

- CUDA binary rebuilds.
- HTTP A/B completes on a clean node.
- The result records whether parity improves.
- Sprint/status/vision are updated with the decision.

## Outcome

Implemented the missing control-stream wait after rank-major router NCCL
allgather in `run_model_router_rank_major_logits`.

Build:

```text
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

passed on gpu-01.

Validation artifact:

```text
/localpool/ds4/workspace/logs/s461-router-wait-graph-gate-s8-t3
```

Shape:

```text
8 requests / 8 slots / 256K context / 3 generated tokens/request
```

| Metric | Control | Graph event-order + router wait |
|---|---:|---:|
| readiness | pass | fail |
| response parity | - | 0/8 |
| HTTP 200 | 8 | 8 |
| server generated decode tok/s | 20.087551 | 9.441529 |
| server continuation decode tok/s | 20.014039 | 9.440907 |
| client generated tok/s | 2.015089 | 0.831814 |
| output-head first token | 52762 | 42549 |
| graph capture attempted/succeeded | 0/0 | 43/43 |
| graph replay attempted/succeeded | 0/0 | 0/0 |
| HC-current gather ms | 4.457466 | 158.533187 |
| HC-current input ms | 224.170869 | 326.915807 |
| min free VRAM MiB | 5092 | 5086 |

## Decision

Do not promote. The router wait is a valid correctness fix, but it is not
sufficient. The graph event-order path remains both incorrect and slower.

## Updated Hypothesis

The root problem is broader than a single missing router allgather edge. The
graph-order path repeatedly reuses shared CUDA events such as `stream_done` and
`dense_done` across many barrier sites inside one decode step. That can create
ambiguous or overwritten dependencies once the host stops synchronizing between
stages. The HC-current gather regression stayed essentially unchanged, which
points back to the event-barrier mechanism itself.

Next work should either:

- allocate distinct per-stage graph-order events for the HC-current stages, or
- add a capture-only mode that leaves eager synchronization intact while
  collecting graph eligibility, so graph capture cannot silently perturb
  serving semantics.
