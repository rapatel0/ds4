# TEMP Status Report 479: SYS Transport Sweep

## Current State

Sprint 479 is complete for the promoted TP/EP serving hot path.

Goal: replace hot-path direct peer-copy transport with non-reducing NCCL
collectives while preserving arithmetic and keeping Direct-SYS peer copies out
of the decode request window.

## Guardrails

- Allowed: `ncclBroadcast` and non-reducing scratch-copy staging.
- Forbidden in this sprint: `ncclAllReduce`, `ncclReduceScatter`, `ncclReduce`,
  or any other reduction replacing a local fixed-order sum.
- The EP local combine kernel remains unchanged.
- This sprint does not use tolerance; it is a transport-only cleanup.

## Implementation

Primary file: `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

Implemented:

- Added NCCL byte broadcast helpers that restore the caller's CUDA device.
- Converted hot GPU0/root fanout to `ncclBroadcast`:
  - HC/current fanout,
  - router and post-attention selected/weight plan uploads,
  - compressed-KV/indexer staging,
  - attention state and sink staging,
  - shared output-head fanout.
- Converted shared-SwiGLU down-input exchange to per-source NCCL broadcast.
- Converted EP compose return transport to broadcast/scratch staging:
  - source rank broadcasts its full contribution buffer,
  - destination ranks locally copy their destination slice from scratch,
  - compact-route variable lengths reuse the existing routed row counts,
  - local fixed-order EP sum is unchanged.
- Removed the unused grouped send/recv helper after validation showed
  non-neighbor V100 pairs require SHM or fail with `NCCL_SHM_DISABLE=1`.

## Validation

Local checks passed:

- `python3 -m py_compile tools/ds4-v100-http-response-tolerance.py
  tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py`
- `bash -n tools/ds4-v100-run-appliance.sh`
- `git diff --check ...`

Remote V100 build passed in `/workspace/ds4-sprint181`:

```bash
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Runtime gates passed:

- `/workspace/s479-transport-smoke-s32-t1-r1j-bcast-ep`
  - `32` slots / `256K` / `1` request.
  - HTTP 200, selected token `48177`, scaffold PASS.
- `/workspace/s479-transport-smoke-s32-t1-r1l-peer-reject`
  - Peer accounting and SYS rejection enabled.
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`, `rejected_requests=0`.
- `/workspace/s479-transport-smoke-s32-t64-r32-peer-reject`
  - `32/32` HTTP 200 responses.
  - All responses emitted `64` tokens.
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`, `rejected_requests=0`.
- `/workspace/s479-transport-smoke-s32-t64-r256-waves-peer-reject`
  - Reference `256` requests / `64` tokens at `32` slots / `256K`.
  - Executed as eight 32-request waves to avoid localhost accept-backlog
    artifacts from 256 simultaneous sockets.
  - `256/256` HTTP 200 responses.
  - `sequence_lengths=64:256`.
  - `peer_copy_ops=0`, `peer_copy_bytes=0`.
  - `peer_copy_sys_ops=0`, `peer_copy_sys_bytes=0`.
  - `rejected_requests=0`.
  - `total_generated_tokens=16384`.
  - `total_continuation_tokens=16128`.
  - `generation_batches=8`, `coalesced_requests=256`.
  - `cumulative_generated_tok_s_decode=35.496696`.
  - `cumulative_continuation_tok_s_decode=35.589837`.
- Post-audit legacy baseline gate validation:
  - V100 build passed for all touched legacy/proxy/workbench/TurboMind targets:
    `tp4-collective-smoke`, `tp4-layer-proxy`, `tp8-collective-smoke`,
    `tp8-collective-workbench`, `tp8-layer-proxy`, `tp8-layer-smoke`,
    `tp8-real-layer-smoke`, `tp8-turbomind-ffn-smoke`, and
    `tp4-turbomind-layer-smoke`.
  - Manual-baseline rejection returned `rc=2` before CUDA work for sampled
    root/doubling/manual modes.
  - `tools/ds4-v100-tp8-layer-proxy` defaulted to `algo=nccl` and passed a
    small NCCL smoke with `verify cross_device_max_abs=0`.
  - `tools/ds4-v100-tp8-collective-workbench` defaulted to `algo=nccl` and
    passed a small all-reduce smoke with `verify max_abs=0`.
  - Additional default-NCCL small smokes passed for `tp4-collective-smoke`,
    `tp4-layer-proxy`, `tp8-collective-smoke`, `tp8-layer-smoke`,
    `tp8-real-layer-smoke`, and `tp4-turbomind-layer-smoke`.
  - `tp8-turbomind-ffn-smoke` builds and gates manual mode, but its current
    correctness fixture fails both manual and NCCL transport reducers on the
    pod, so it remains non-promotional evidence.
  - `kernels/turbomind/ggml-turbomind/test_tp4_resident_layer_slice.cu`
    defaults to NCCL all-reduce. Direct `nvcc` validation passed on the V100
    pod (`tpa=1`, one layer, one iteration, correctness PASS); explicit
    `DS4_TP4_RESIDENT_ALGO=root` rejects unless
    `DS4_ALLOW_MANUAL_PEER_BASELINE=1`.
  - `kernels/turbomind/ggml-turbomind/test_tp_split_2gpu.cpp` and
    `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp` now default the
    copy-inclusive proxy transport to NCCL. Input distribution uses NCCL
    broadcast, output collection uses grouped NCCL send/recv, and the old
    direct peer transport requires
    `DS4_TP_SPLIT_TRANSPORT=peer DS4_ALLOW_MANUAL_PEER_BASELINE=1`. Direct
    `nvcc` builds passed; one-case NCCL smokes passed correctness for both
    proxies.

## Residuals

Static grep still finds direct-copy wrapper definitions and cold/diagnostic call
sites under graph-copy wrappers. The standalone output-head resident gate now
uses a local NCCL communicator for its replicated input broadcast. The remaining
graph-wrapper sites are not in the promoted TP/EP serving path validated above;
the peer-reject reference run would have rejected any hot Direct-SYS direct
copy.

Post-audit update: the `--decode-cudagraph-peer-copy-gate` diagnostic is now
rejected unconditionally and the direct peer-copy implementation was removed
from the TP/EP serving binary. Static grep now finds no `cudaMemcpyPeerAsync`,
`cudaMemcpyPeer`, or `ds4_peer_copy_async` in
`tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

Legacy/proxy/workbench follow-up: `tools/ds4-v100-tp8-layer-proxy.cu` and
`tools/ds4-v100-tp8-collective-workbench.cu` default to `--algo nccl`; their
root/doubling direct peer-copy algorithms require
`--allow-manual-peer-baseline`. TP4/TP8 collective and layer smoke tools now
also default to NCCL all-reduce. TurboMind TP4/TP8 smoke tools default to NCCL
broadcast-to-root transport plus their existing fixed-order float accumulation;
their old synchronous `cudaMemcpyPeer` reducer requires
`--reduce-algo manual --allow-manual-peer-baseline`.

TurboMind split proxy follow-up: the 2-GPU and 4-GPU copy-inclusive proxies now
default to NCCL transport. Their remaining raw peer-copy branches are explicit
manual baselines gated by `DS4_ALLOW_MANUAL_PEER_BASELINE=1`.

Static classification after the follow-up: remaining raw peer-copy findings are
generic tensor/binding copy helpers, the explicit P2P bandwidth proxy, or
manual-baseline branches in legacy diagnostic tools. They are not promoted
TP/EP appliance broadcast or reduction paths.

Keep `DS4_V100_TP_EP_PEER_REJECT_SYS=0` by default for now, and use it as the
validation gate when changing transport code.

## Decision

Promote Sprint 479's hot-path transport cleanup. Direct-SYS peer-copy traffic is
zero in the validated reference serving window, and no arithmetic-changing
reduction was introduced.

## Post-Sprint Reduction Audit

The separate NCCL reduction audit evaluated EP compose `ncclReduceScatter`
under the reduced tolerance gate. This was not part of Sprint 479's
transport-only promotion because it changes fp32 reduction order.

Result:

- Artifact: `/workspace/s480-ep-reducescatter-tolerance`
- Local tolerance summary:
  `/tmp/s480-ep-reducescatter-tolerance/response-tolerance.json`
- Shape: `32` slots / `256K` / `32` paired selected-token requests.
- Control: non-compact FP32 compose, `compose_reduce_scatter=0`, `32/32` HTTP.
- Candidate: non-compact FP32 compose, `compose_reduce_scatter=1`, `32/32`
  HTTP.
- selected-token agreement: `32/32 = 1.0`.
- generated-sequence agreement: `1.0`.
- max selected-logit relative error: `7.054008547965787e-05`.
- threshold: `1e-3`.
- tolerance pass: `true`.
- Short-run compose metrics:
  - projected slot-step tok/s: `29.442507 -> 32.059331`,
  - compose ms: `40.540996 -> 11.648631`,
  - compose copy ms: `34.540488 -> 0.000000`.

Decision: the existing `--nccl-reduce-scatter-compose-gate` is tolerance-cleared
for the compatible non-compact FP32 compose path. It is not a new default for
the promoted compact-route serving profile: compact-route compose deliberately
bypasses dense reduce-scatter, and launcher defaults still keep compact route
enabled.
