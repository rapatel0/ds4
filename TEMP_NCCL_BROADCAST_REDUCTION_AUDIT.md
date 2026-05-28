# TEMP NCCL Broadcast/Reduction Audit

Current state after Sprint 479.

## Scope

Objective: every TP/EP appliance broadcast or reduction should use NCCL when it
is in the serving path. Pure communication replacements are exact-gated;
reductions that change arithmetic order are tolerance-gated.

## Serving Path

Promoted TP/EP serving hot path:

- GPU0/root fanout uses `ncclBroadcast` helpers.
- Shared-SWiGLU down-input fanout uses per-source `ncclBroadcast`.
- EP compose return transport uses NCCL broadcast/scratch staging and preserves
  the existing fixed-order local sum.
- Standalone output-head resident diagnostic input fanout now uses a local NCCL
  communicator.
- Existing NCCL reductions/gathers are gated:
  - `--tp-hc-current-allreduce-gate`
  - `--model-router-allreduce-logits-gate`
  - `--nccl-reduce-scatter-compose-gate`
  - `--true-ds4-attention-output-nccl-allgather-gate`
  - `--tp-hc-current-input-nccl-allgather-gate`

Validation artifact:

- `/workspace/s479-transport-smoke-s32-t64-r256-waves-peer-reject`
  - `256/256` HTTP 200
  - `sequence_lengths=64:256`
  - `peer_copy_ops=0`
  - `peer_copy_sys_bytes=0`
  - `rejected_requests=0`

## Exact-Gated Communication

Transport-only changes must preserve arithmetic and selected-token behavior.

Current exact gate:

- Enable peer accounting and SYS rejection.
- Run the reference selected-token window.
- Require `peer_copy_ops=0` and `peer_copy_sys_bytes=0` in the serving window.
- Require response success and selected-token/sequence agreement for A/B
  transport candidates.

Retired diagnostic:

- `--decode-cudagraph-peer-copy-gate` is now rejected unconditionally. The old
  diagnostic branch used direct peer-copy graph transport; the TP/EP serving
  binary now keeps graph/cold copy helpers on the default copy-kernel path or
  the NCCL broadcast/reduction paths above.

## Tolerance-Gated Reductions

Reduction changes can reorder fp32 summation and must not use the transport-only
bit-exact gate.

Use the relaxed reduced tolerance checker:

- selected-token agreement >= `0.99`
- generated-sequence agreement >= `0.99`
- max selected-logit relative error is advisory only

Known reduction candidates:

- A3 router logits all-reduce: promoted under the relaxed agreement-only policy
  in Sprint 480. Existing post-479 A/B artifact
  `/workspace/s480-a3-router-allreduce-tolerance` served both legs `32/32` at
  `32` slots / `256K`, and selected-token / generated-sequence agreement was
  `1.0`. Max selected-logit relative error was `0.025157711827123192`,
  reported as advisory only. The appliance default now enables
  `DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=1`.
- A6 rank-local attention projection input: evaluated and rejected at the
  relaxed agreement gate with selected-token / generated-sequence agreement
  `0.03125`.
- EP compose `ncclReduceScatter`: promoted for the compatible non-compact FP32
  compose path. Artifact
  `/workspace/s480-ep-reducescatter-tolerance` served both legs `32/32` at
  `32` slots / `256K`, with `compose_reduce_scatter=1` in the candidate.
  Local summary
  `/tmp/s480-ep-reducescatter-tolerance/response-tolerance.json` reported
  selected-token agreement `1.0`, generated-sequence agreement `1.0`, and max
  selected-logit relative error `7.054008547965787e-05` advisory only. The
  short run improved projected slot-step tok/s
  `29.442507 -> 32.059331` and reduced compose time
  `40.540996 -> 11.648631` by replacing compose copy
  `34.540488 -> 0.000000`.
  Decision: the launcher default is now `auto`, which enables this gate for
  non-compact FP32 compose and disables it for compact-route serving. The
  promoted compact-route path still bypasses reduce-scatter by design because
  route-indexed compose is not a dense reduce-scatter shape.
- Final LM-head/local argmax reductions and other future arithmetic reductions
  should use the same tolerance policy.

## Remaining Direct Peer Copies

Static direct-copy findings outside the promoted serving hot path:

- No `cudaMemcpyPeerAsync`, `cudaMemcpyPeer`, or `ds4_peer_copy_async` remains
  in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Graph-copy wrapper call sites remain, but they now use copy kernels unless a
  surrounding path has been converted to NCCL broadcast/reduction.
- V100 build passed after retiring the direct peer-copy diagnostic. A negative
  flag check exits `rc=2` with the message that
  `--decode-cudagraph-peer-copy-gate` has been retired.
- Legacy/proxy/workbench tools such as `tools/ds4-v100-tp8-layer-proxy.cu`,
  `tools/ds4-v100-tp8-collective-workbench.cu`, and older TP4/TP8 smoke tools.

Post-audit baseline gating update:

- `tools/ds4-v100-tp8-layer-proxy.cu` and
  `tools/ds4-v100-tp8-collective-workbench.cu` now default to `--algo nccl`.
  Their root/doubling direct peer-copy algorithms require
  `--allow-manual-peer-baseline`.
- The straightforward TP4/TP8 collective and layer smoke tools now have NCCL
  as their normal path:
  - `tools/ds4-v100-tp4-collective-smoke.cu`
  - `tools/ds4-v100-tp4-layer-proxy.cu`
  - `tools/ds4-v100-tp8-collective-smoke.cu`
  - `tools/ds4-v100-tp8-layer-smoke.cu`
  - `tools/ds4-v100-tp8-real-layer-smoke.cu`
  Their root/doubling direct peer-copy algorithms require
  `--allow-manual-peer-baseline`.
- TurboMind TP4/TP8 smoke tools default to NCCL broadcast-to-root transport
  plus the existing fixed-order float accumulation. Their old synchronous
  `cudaMemcpyPeer` root reducers require
  `--reduce-algo manual --allow-manual-peer-baseline`:
  - `tools/ds4-v100-tp4-turbomind-layer-smoke.cu`
  - `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu`
- V100 validation passed after this update:
  - all touched legacy/proxy/workbench/TurboMind targets built with
    `CUDA_ARCH=sm_70`;
  - sampled default baseline invocations exited `rc=2` with the manual-baseline
    message before CUDA work;
  - NCCL-capable defaults passed small NCCL smoke checks with zero verification
    error.
  - `tools/ds4-v100-tp4-turbomind-layer-smoke` passed with
    `reduce_algo=nccl` against `build/turbomind-v100/libggml-turbomind.so`.
- `tools/ds4-v100-tp8-turbomind-ffn-smoke` builds and gates manual mode, but
  its current correctness fixture fails both the old manual reducer and the
  NCCL transport reducer on the pod; do not use it as promotion evidence
  until that fixture is repaired independently.
- `kernels/turbomind/ggml-turbomind/test_tp4_resident_layer_slice.cu` now
  defaults `DS4_TP4_RESIDENT_ALGO` to `nccl` and uses `ncclAllReduce` for the
  resident hidden reduction. The old root/doubling peer-copy algorithms require
  `DS4_ALLOW_MANUAL_PEER_BASELINE=1`. Direct `nvcc` validation passed on the
  V100 pod because that image lacks `/usr/bin/cmake`: the `tpa=1`, one-layer,
  one-iteration NCCL smoke passed correctness, and an explicit
  `DS4_TP4_RESIDENT_ALGO=root` run rejected without the manual-baseline env.
- `kernels/turbomind/ggml-turbomind/test_tp_split_2gpu.cpp` and
  `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp` now default
  their copy-inclusive proxy transport to `DS4_TP_SPLIT_TRANSPORT=nccl`:
  input distribution uses NCCL broadcast and output collection uses grouped
  NCCL send/recv. The old direct peer-copy transport requires
  `DS4_TP_SPLIT_TRANSPORT=peer DS4_ALLOW_MANUAL_PEER_BASELINE=1`.
  Direct `nvcc` builds passed on the V100 pod. One-case NCCL smokes passed:
  `tp_split_2gpu` reported `transport=nccl` and correctness PASS with
  max_abs `4.5776e-05`; `tp_split_4gpu` reported `transport=nccl` and
  correctness PASS with max_abs `4.0054e-05`. Explicit peer transport without
  the allow env rejected with the manual-baseline message.

Codebase-wide static classification after the follow-up:

- Generic copy APIs remain in `ds4_cuda.cu`, `ds4_v100_context_cuda.cu`, and
  TurboMind Python bindings. These are not TP/EP appliance broadcast or
  reduction collectives.
- `kernels/turbomind/ggml-turbomind/test_p2p_reduce_proxy.cpp` is explicitly a
  peer-copy bandwidth proxy, not a promoted collective path.
- Remaining raw peer-copy sites in legacy tools and TurboMind fixtures are
  manual-baseline branches behind explicit flags/env vars; their normal paths
  are NCCL where the site is a broadcast or reduction proxy.

Decision:

- Keep legacy/proxy/workbench manual P2P algorithms as explicit baselines
  because those tools compare root/doubling manual collectives against NCCL.
- Do not use those manual P2P modes as appliance promotion evidence.
- For appliance serving validation, peer SYS rejection plus NCCL graph SYS
  checks remain the authority; after retiring the graph peer-copy diagnostic,
  the expected peer-copy counts in the promoted TP/EP serving binary are zero.

## Next Work

1. Keep A3 promoted under the relaxed agreement-only policy and keep A6
   rejected until the rank-local norm divergence is fixed and re-evaluated as a
   fresh candidate.
2. If legacy/proxy/workbench tools are ever used as promotion evidence, run
   their existing NCCL modes rather than the manual root/doubling baselines.
