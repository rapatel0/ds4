# Sprint 479 Intent: SYS Transport Sweep

## Seed Prompt

Review `TEMP_SYS_TRANSPORT_SWEEP.md`, create a sprint plan, then execute it.
The sprint is self-contained: replace hot-path direct peer-copy transport with
non-reducing NCCL collectives while preserving bit-exact arithmetic.

## Orientation Summary

- Sprint 478 completed the HC-current A2 all-reduce, A4a full-current NCCL
  cleanup, A3 default-off router all-reduce, and the reduced arithmetic
  tolerance gate. A6 was evaluated and rejected by tolerance, so Sprint 479 must
  stay transport-only and bit-exact.
- The active file is `tools/ds4-v100-tp-ep-full-layer-smoke.cu`; launcher and
  profile harnesses already initialize `r.compose_nccl` for the promoted TP/EP
  path.
- Remaining hot transport surfaces are direct `ds4_peer_copy_async` calls and
  graph-copy wrappers in GPU0 broadcasts, router-plan upload, shared FFN
  exchange, EP dispatch/combine, compressed/indexer staging, and attention
  sinks.
- Guardrail: use only non-reducing NCCL collectives (`ncclBroadcast`,
  `ncclAllToAll`, `ncclSend`, `ncclRecv`). Do not replace fixed-order local
  summation with NCCL reductions.
- Verification must be strict selected-token parity plus peer-accounting SYS
  elimination. No tolerance gate applies to this sprint.

## Relevant Code Areas

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  - `ds4_peer_copy_async_impl`
  - graph copy helpers around `enqueue_graph_f32_copy_*`
  - HC-current/full-current staging
  - route-plan upload
  - shared FFN down-input materialization
  - EP compose copy paths
  - compressed-KV/indexer staging
  - attention sink staging
- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`
- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`

## Success Criteria

- All targeted hot-path direct peer copies are replaced by non-reducing NCCL
  transport.
- Local and V100 pod builds pass.
- Reference selected-token parity is bit-exact.
- Peer accounting reports zero SYS bytes at replaced sites and near-zero
  aggregate Direct-SYS bytes.
- Request-window decode throughput and GPU utilization do not regress versus
  control.

## Verification Strategy

- Static grep for remaining `ds4_peer_copy_async` and graph-copy wrapper usage,
  with any intentionally retained cold or non-hot usage documented.
- Remote build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`.
- Short smoke on V100 before long gate.
- Reference selected-token gate at 32 slots / 256K / 256 requests / 64 tokens.
- Peer-accounting run with per-site SYS report.

## Uncertainty

- Correctness: Medium. The intended replacements are non-reducing, but EP sparse
  routing and graph-order sites can accidentally change ordering if implemented
  too aggressively.
- Scope: High. The current file has many direct-copy sites and some are in
  branches that are diagnostic-only.
- Architecture: Medium. The safest first pass is helper-based NCCL transport,
  followed by targeted conversion of EP pairwise exchange.

## Open Questions

- Which remaining `ds4_peer_copy_async` sites are truly hot after Sprint 478 and
  which can be left as cold diagnostics with explicit documentation?
- Does the installed NCCL version expose `ncclAllToAll`; if not, dense uniform
  all-to-all must use grouped `ncclSend`/`ncclRecv`.
- Can graph-copy wrappers be fully converted this sprint without disturbing the
  still-default-off graph replay work?
