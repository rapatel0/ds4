# Sprint 479: SYS Transport Sweep

## Overview

Replace hot-path direct peer-copy transport in the TP/EP serving binary with
non-reducing NCCL collectives. This is a transport-only sweep: bytes and local
arithmetic stay the same, but cross-rank movement must route through NCCL so the
V100 topology avoids SYS/PCIe paths.

## Use Cases

- As an operator, I can run the 32-slot / 256K TP/EP serving path without
  Direct-SYS peer-copy traffic in the hot decode window.
- As an implementer, I can distinguish transport-only NCCL swaps from
  arithmetic-changing reductions and keep the strict bit-exact gate.
- As a future graph-sprint implementer, I inherit capturable NCCL transport
  rather than uncapturable peer-copy wrappers.

## Architecture

Only non-reducing NCCL collectives are allowed:

- `ncclBroadcast` for GPU0-sourced replicated buffers.
- `ncclAllToAll` for uniform per-rank dense pairwise exchange when available.
- grouped `ncclSend` / `ncclRecv` for sparse or variable per-pair exchange.

Do not replace the fixed-order local EP combine kernel with `ncclReduceScatter`,
`ncclAllReduce`, or any other reducing collective. This sprint must remain
bit-exact.

All new collectives use each rank's existing `r.compose_nccl` communicator and
rank stream, grouped with `ncclGroupStart/End` when issuing across ranks.

## Implementation

1. Add reusable NCCL transport helpers in
   `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:
   - GPU0 float/int broadcast helpers,
   - uniform all-to-all or send/recv helper,
   - variable send/recv helper for compact route exchange.
2. Replace hot GPU0 broadcasts:
   - HC split and current/full-current staging still left after Sprint 478,
   - router selected/weights plan upload,
   - compressed-KV/indexer staging,
   - attention sink staging,
   - input/embedding and output-head hot fanout if present.
3. Replace EP/shared pairwise transport:
   - dense uniform EP exchange with non-reducing NCCL all-to-all/send-recv,
   - compact-route exchange with grouped send/recv using existing active route
     counts,
   - shared FFN down-input materialization pairwise copies.
4. Convert graph-copy wrapper call sites where the wrapped transport is one of
   the hot patterns above.
5. Keep arithmetic and local combine kernels unchanged.
6. Update launcher/profile/harness notes only if new gates or telemetry fields
   are needed. Prefer no new production default gate because the sweep is a
   transport-only cleanup.

## Files Summary

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`: primary implementation.
- `docs/sprints/SPRINT-479.md`: sprint record and outcomes.
- `TEMP_STATUS_REPORT_479.md`: short handoff/status report.
- `docs/sprints/VISION.md`: final outcome row after validation.

## Definition Of Done

- Local syntax/static checks pass.
- V100 pod build passes with `CUDA_ARCH=sm_70`.
- Static grep documents remaining direct peer-copy sites; no hot-path target is
  left without an explanation.
- Short selected-token smoke passes.
- Reference selected-token parity is strict bit-exact at 32 slots / 256K /
  256 requests / 64 tokens.
- Peer accounting reports per-site SYS bytes of zero for replaced sites and
  aggregate Direct-SYS bytes approximately zero in the decode window.
- Decode tok/s and request-window GPU utilization are at or above control.
- Outcome is documented with promote/reject decision and updated residual
  bottleneck/domain table.

## Risks

- Accidentally using reducing NCCL collectives would change fp32 summation order.
- Sparse route exchange can use variable counts; treating it as uniform all-to-all
  can be wrong or wasteful.
- Some graph-copy wrappers may sit in diagnostic graph paths rather than the
  promoted serving path; conversion must not disturb graph-order correctness.
- The pod image lacks Python/curl, so direct validation may need launcher plus
  `/dev/tcp` or local harness orchestration.

## Security

No external service exposure changes. The sprint only changes intra-node GPU
transport paths and local validation artifacts.

## Dependencies

- Existing `r.compose_nccl` communicator initialization.
- V100 pod `llm/ds4-tp-bench`.
- Peer-copy accounting from Sprint 476/478.

## Open Questions

- If `ncclAllToAll` is not available in the installed NCCL headers, should the
  dense uniform path use grouped send/recv for this sprint and leave native
  all-to-all as a later cleanup?
- Should any cold diagnostic direct-copy sites be preserved intentionally, or
  should the static grep target be zero raw uses outside the accounting wrapper?

## Execution Update

Status: complete for the promoted TP/EP serving hot path.

Completed:

- Added reusable NCCL broadcast helpers in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`; helper calls restore the caller's
  current CUDA device.
- Converted hot GPU0/root fanout sites to `ncclBroadcast`, including
  HC/current fanout, router/post-attention route metadata, compressed-KV and
  indexer staging, attention state fanout, attention sink staging, and shared
  output-head fanout.
- Converted shared-SWiGLU down-input exchange from pairwise direct copies to
  per-source `ncclBroadcast`; each source shard is replicated to every
  destination rank at the source-specific offset.
- Replaced EP compose's destination-specific pairwise direct copies with
  broadcast/scratch staging:
  - each source rank broadcasts its full contribution buffer over NCCL,
  - each destination rank locally copies the destination slice from its scratch
    buffer into the existing `d_ep_remote[src]` slot,
  - compact-route variable copy lengths still use the existing
    `routed_compose_rows`/`copy_elems_by_src` counts,
  - the downstream fixed-order local EP sum kernel is unchanged.
- Removed the attempted grouped send/recv helper after confirming that
  non-neighbor V100 pairs fall back to SHM or fail with `NCCL_SHM_DISABLE=1`.

Validation:

- Local syntax/static checks passed:
  - `python3 -m py_compile tools/ds4-v100-http-response-tolerance.py
    tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py`
  - `bash -n tools/ds4-v100-run-appliance.sh`
  - `git diff --check` for the sprint-touched files.
- Remote V100 build passed in `/workspace/ds4-sprint181` with:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70
  tools/ds4-v100-tp-ep-full-layer-smoke`.
- Short selected-token smoke passed:
  - artifact `/workspace/s479-transport-smoke-s32-t1-r1j-bcast-ep`
  - `32` slots / `256K` / `1` request,
  - HTTP 200, selected token `48177`, full scaffold PASS,
  - `compose_copy_ms=36.532830`.
- Peer-reject selected-token smoke passed:
  - artifact `/workspace/s479-transport-smoke-s32-t1-r1l-peer-reject`
  - `peer_copy_accounting=1`, `peer_copy_reject_sys=1`,
  - `peer_copy_ops=0`, `peer_copy_bytes=0`, `peer_copy_sys_ops=0`,
    `peer_copy_sys_bytes=0`,
  - `rejected_requests=0`, `total_generated_tokens=1`.
- 32-request continuation peer-reject run passed:
  - artifact `/workspace/s479-transport-smoke-s32-t64-r32-peer-reject`
  - `32/32` HTTP 200 responses,
  - all responses emitted `64` generated tokens,
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`,
  - `total_generated_tokens=2048`, `total_continuation_tokens=2016`.
- Reference 256-request / 64-token peer-reject run passed using eight
  32-request waves to avoid localhost accept-backlog artifacts:
  - artifact `/workspace/s479-transport-smoke-s32-t64-r256-waves-peer-reject`
  - `256/256` HTTP 200 responses,
  - all responses emitted `64` generated tokens,
  - `generation_batches=8`, `coalesced_requests=256`,
  - `peer_copy_ops=0`, `peer_copy_bytes=0`, `peer_copy_sys_ops=0`,
    `peer_copy_sys_bytes=0`,
  - `rejected_requests=0`,
  - `total_generated_tokens=16384`,
    `total_continuation_tokens=16128`,
  - `cumulative_generated_tok_s_decode=35.496696`,
    `cumulative_continuation_tok_s_decode=35.589837`.
- Post-audit legacy/proxy/workbench validation passed:
  - V100 build passed for all touched targets:
    `tools/ds4-v100-tp4-collective-smoke`,
    `tools/ds4-v100-tp4-layer-proxy`,
    `tools/ds4-v100-tp8-collective-smoke`,
    `tools/ds4-v100-tp8-collective-workbench`,
    `tools/ds4-v100-tp8-layer-proxy`,
    `tools/ds4-v100-tp8-layer-smoke`,
    `tools/ds4-v100-tp8-real-layer-smoke`,
    `tools/ds4-v100-tp8-turbomind-ffn-smoke`, and
    `tools/ds4-v100-tp4-turbomind-layer-smoke`.
  - Sampled root/doubling/manual modes reject by default with `rc=2` before
    CUDA work unless the manual-baseline flag is passed.
  - NCCL-capable defaults passed small NCCL smoke checks:
    `tools/ds4-v100-tp8-layer-proxy` reported `algo=nccl` and
    `verify cross_device_max_abs=0`; `tools/ds4-v100-tp8-collective-workbench`
    reported `algo=nccl` and `verify max_abs=0`.
  - Expanded NCCL-default smokes also passed for the TP4/TP8 collective/layer
    tools and `tools/ds4-v100-tp4-turbomind-layer-smoke`.
  - `tools/ds4-v100-tp8-turbomind-ffn-smoke` builds and gates manual mode, but
    its current pod fixture fails correctness with both the old manual reducer
    and the NCCL transport reducer; keep it out of promotion evidence until
    that fixture is repaired.
  - `kernels/turbomind/ggml-turbomind/test_tp4_resident_layer_slice.cu`
    defaults to NCCL all-reduce and rejects root/doubling manual peer-copy
    algorithms unless `DS4_ALLOW_MANUAL_PEER_BASELINE=1` is set. Direct `nvcc`
    validation passed on the V100 pod because the image lacks CMake.
  - `kernels/turbomind/ggml-turbomind/test_tp_split_2gpu.cpp` and
    `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp` now default the
    copy-inclusive split-proxy transport to NCCL. Input distribution uses
    NCCL broadcast; output collection uses grouped NCCL send/recv. The old
    direct peer-copy transport requires
    `DS4_TP_SPLIT_TRANSPORT=peer DS4_ALLOW_MANUAL_PEER_BASELINE=1`.
    Direct `nvcc` builds passed on the V100 pod, and one-case NCCL smokes
    passed correctness for both proxies.

Residual static sites:

- `ds4_peer_copy_async_impl` and the graph-copy wrapper definitions remain so
  peer accounting and diagnostic graph paths still compile.
- Static call sites remain under graph/cold diagnostic paths. The standalone
  output-head resident gate now uses its own local NCCL communicator for
  replicated input broadcast. The remaining graph-wrapper sites are not
  exercised by the promoted TP/EP serving path validated above: the peer-reject
  run would have rejected any hot Direct-SYS direct copy.

Post-audit update:

- `--decode-cudagraph-peer-copy-gate` is now rejected unconditionally and the
  direct peer-copy implementation was removed from the TP/EP serving binary.
- Static grep now finds no `cudaMemcpyPeerAsync`, `cudaMemcpyPeer`, or
  `ds4_peer_copy_async` in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Graph-copy wrapper call sites remain, but they use copy kernels unless their
  surrounding path is on a NCCL broadcast/reduction branch.
- Legacy/proxy/workbench follow-up:
  - `tools/ds4-v100-tp8-layer-proxy.cu` and
    `tools/ds4-v100-tp8-collective-workbench.cu` now default to `--algo nccl`.
    Their root/doubling direct peer-copy algorithms require
    `--allow-manual-peer-baseline`.
  - TP4/TP8 collective/layer smoke tools now default to NCCL all-reduce:
    `tools/ds4-v100-tp4-collective-smoke`,
    `tools/ds4-v100-tp4-layer-proxy`,
    `tools/ds4-v100-tp8-collective-smoke`,
    `tools/ds4-v100-tp8-layer-smoke`, and
    `tools/ds4-v100-tp8-real-layer-smoke`.
  - TurboMind TP4/TP8 smoke tools default to NCCL broadcast-to-root transport
    plus the existing fixed-order float accumulation; their old synchronous
    `cudaMemcpyPeer` reducer requires
    `--reduce-algo manual --allow-manual-peer-baseline`.
  - TurboMind split proxies default to NCCL transport for broadcast/gather
    proxy movement. Their old peer-copy transport is an explicit manual
    baseline behind `DS4_ALLOW_MANUAL_PEER_BASELINE=1`.
  - Remaining raw peer-copy static findings are classified as generic tensor
    or Python binding copy helpers, explicit P2P bandwidth proxies, or gated
    manual-baseline branches in legacy diagnostics. No promoted TP/EP appliance
    broadcast or reduction path remains on raw peer-copy.

Decision:

- Promote the Sprint 479 transport cleanup for the TP/EP serving hot path.
- Keep peer SYS rejection as a validation gate, not a launcher default, because
  legacy/proxy/workbench tools still contain manual P2P baselines outside the
  TP/EP serving binary.
- Do not promote any reducing collective as part of this sprint; EP compose
  arithmetic stayed in the existing fixed-order local kernel.

Post-sprint reduction audit:

- EP compose `ncclReduceScatter` was evaluated separately under the Sprint 478
  reduced tolerance checker, because it changes fp32 reduction order and was
  intentionally out of scope for this transport-only sprint.
- Artifact: `/workspace/s480-ep-reducescatter-tolerance`.
- Shape: `32` slots / `256K` / `32` paired selected-token requests.
- Candidate startup confirmed non-compact FP32 compose with
  `compose_reduce_scatter=1`; control used the same non-compact path with
  `compose_reduce_scatter=0`.
- Both legs served `32/32` HTTP responses.
- Tolerance result:
  - selected-token agreement `1.0`,
  - generated-sequence agreement `1.0`,
  - max selected-logit relative error `7.054008547965787e-05` versus `1e-3`,
  - pass `true`.
- Short-run compose metrics improved in the non-compact path:
  - projected slot-step tok/s `29.442507 -> 32.059331`,
  - compose ms `40.540996 -> 11.648631`,
  - compose copy ms `34.540488 -> 0.000000`.

Decision: the existing `--nccl-reduce-scatter-compose-gate` is tolerance-cleared
for compatible non-compact FP32 compose. This is not a compact-route serving
default promotion; compact-route compose still bypasses dense reduce-scatter by
design.
