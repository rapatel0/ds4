---
sprint: 308
title: TP/EP HC Semantic Parity
status: in_progress
started: 2026-05-24
branch: claude-takeover
---

# Sprint 308 - TP/EP HC Semantic Parity

## Goal

Close the parity gap exposed by Sprint 307 before treating the TP/EP HTTP path
as production-serving code.

## Initial Finding

The Sprint 307 reference harness failed `short_reasoning_plain`: the official
selected-token bytes are `3136` (`16`), while the TP/EP HTTP path returned
`494343` (`ICC`).

The first audit shows this is not an OpenAI API envelope or tokenizer issue.
The current TP/EP resident layer path still contains diagnostic semantics:

- EP routing is synthetic: `build_offsets_for_rank()` assigns routes from slot
  and rank arithmetic instead of the model router.
- Resident expert tables packed only six local experts per GPU, which is
  incompatible with true router-driven EP over `256` global experts.
- The attention body is still a bridge around simplified dense paths rather
  than the full DS4 Q/KV/RoPE/compressed-attention/indexer sequence.

## Work Started

- Removed the six-local-expert diagnostic cap in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- TP/EP now attempts to resident-pack all `32` local experts per GPU.
- Route buffers now allocate for worst-case `slots * top_k` routes per rank
  instead of the synthetic route count. This is required before data-driven
  routing, because a real router can send more routes to one GPU than the
  balanced synthetic schedule.
- Routed expert contributions now flow through a per-route weight buffer.
  The synthetic schedule uses `0.125` route weights to preserve current
  behavior, but compose no longer owns a hardcoded EP scaling factor. Real
  router weights can now be uploaded with the route plan.
- Added opt-in model-router route selection for the TP/EP HTTP path:
  `DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1` enables loading
  `ffn_gate_inp.weight`, optional `exp_probs_b`, and optional
  `ffn_gate_tid2eid` hash metadata, computes router logits from the
  FFN-normalized HC current vector, and uploads selected expert IDs plus
  per-route weights into the EP route plan.
- Added explicit active-slot masking for model-router routes. Token ID `0` is
  no longer treated as inactive; the HTTP scheduler now passes a separate
  active-slot vector for prefill and decode. This matters for production
  serving because token IDs and cache-slot occupancy are different concepts.
- Added launcher support for the model-router route mode. The launcher forces
  HC-current input and final-HC expansion, requires `top_k=6`, and rejects
  compact route compose because true model routing can produce multiple
  selected experts on the same rank for one slot.

## Validation Plan

1. Build the updated TP/EP binary on the V100 pod. Complete.
2. Start the TP/EP server with the same 32-slot / 256K launcher settings used
   in Sprint 307. Complete.
3. Confirm all-local-expert residency fits in 32GB V100 VRAM. Complete.
4. Re-run the Sprint 307 reference harness and record whether the result
   changes. Complete.
5. Implement router-driven EP routing next. Correctness can start with a
   host-rebuilt route table per step; device-side routing can follow after the
   parity signal improves.

## V100 Result

The updated binary builds on `llamacpp-build-8gpu` and the all-local-expert
resident load fits:

- Expert bindings: `147169738752` bytes aggregate,
  `18396217344` bytes per GPU.
- TP runtime memory budget: `7122628608` bytes per GPU for KV/state/scratch.
- Dense F16 cache: `14451998720` bytes aggregate.
- Observed first-run loaded GPU memory was about `27.3 GiB` per V100, within
  the 32GB cards.

The parity result did not change after all-local-expert residency:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "ICC",
  "generated_token_sequence": [95933],
  "wall_tok_s": 193.461777,
  "decode_tok_s": 306.956969
}
```

This localizes the current production blocker: full expert residency is
necessary but not sufficient. The next implementation step is true
router-driven EP scheduling; after that, the remaining attention bridge can be
isolated with layer-stage parity checks.

The follow-up route-capacity build also passed on the V100 pod and preserved
the same expected failure:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "ICC",
  "generated_token_sequence": [95933],
  "wall_tok_s": 192.892092,
  "decode_tok_s": 307.145279
}
```

The weighted-route build also passed on the V100 pod and preserved the same
expected failure:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "ICC",
  "generated_token_sequence": [95933],
  "wall_tok_s": 194.878813,
  "decode_tok_s": 304.803471
}
```

The first model-router top-k run was stable but still failed parity:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "尷",
  "generated_token_sequence": [117465],
  "wall_tok_s": 158.650000,
  "decode_tok_s": 231.800000
}
```

Adding hash-router metadata changed the wrong token but did not close parity:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "{Doxy",
  "generated_token_sequence": [85766],
  "wall_tok_s": 161.740000,
  "decode_tok_s": 231.760000
}
```

Feeding the routed expert path from `ffn_normed` exposed a real numeric
stability blocker. An explicit opt-in gate,
`DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1`, now switches only the routed expert
pack source to `ffn_normed` while leaving attention/shared bridge inputs on
the stable raw HC-current path. On the V100 pod this fails immediately at
layer `0` with `decode_finite_bad=16384`, `rc=5`, and EP time jumping to
about `50.6 ms`. That localizes the next bug to the normalized routed expert
input/TurboMind interaction rather than later attention.

Until this is fixed, the current stable mode uses FFN-normalized HC for
router logits while keeping routed expert input on the raw HC-current bridge.

The active-slot-mask rerun confirms that real active HTTP slots now produce
nonzero model-router routes:

```text
layer 0 routes 0,2,0,0,0,0,2,2
layer 1 routes 1,0,0,1,2,2,0,0
layer 2 routes 0,2,0,2,2,0,0,0
layer 3 routes 0,1,0,2,3,0,0,0
layer 4 routes 1,1,0,1,1,1,1,0
layer 5 routes 1,1,1,1,1,0,1,0
```

The same reference vector still fails:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": " ICC",
  "generated_token_sequence": [61317],
  "wall_tok_s": 164.721272,
  "decode_tok_s": 237.349475
}
```

This is progress but not production correctness. The route path is no longer
synthetic/no-op for active slots; the remaining mismatch is now in the layer
semantics still being bridged: full shared FFN, normalized routed-expert input,
and full DS4 attention/compressed-KV/indexer math.

The routed-FFN-normalized-input stats run narrows the immediate numeric
failure further:

```text
route_input layer 0 rank 1 finite_bad=0 max_abs=38.53125
route_input layer 0 rank 6 finite_bad=0 max_abs=38.53125
route_input layer 0 rank 7 finite_bad=0 max_abs=38.53125
route_gated layer 0 rank 1 finite_bad=0 max_abs=0
route_down  layer 0 rank 1 finite_bad=0 max_abs=0
route_gated layer 0 rank 6 finite_bad=0 max_abs=0
route_down  layer 0 rank 6 finite_bad=0 max_abs=0
route_gated layer 0 rank 7 finite_bad=1 max_abs=46176
route_down  layer 0 rank 7 finite_bad=4096 max_abs=1375
```

So the `ffn_normed` route input itself is finite. The failure appears inside
the routed expert invocation/table/scale path for the rank-7 selected experts,
while ranks 1 and 6 unexpectedly produce all-zero expert outputs. The next
debug step should print selected global/local expert IDs and inspect the
rank-local TurboMind pointer/scale bindings for those experts.

The follow-up route-ID trace identified the exact layer-0 routes:

```text
slot 0 k0 expert 254 rank 7 local 30 weight 5.48186296e-09
slot 0 k1 expert 222 rank 6 local 30 weight 0.000506118406
slot 0 k2 expert 245 rank 7 local 21 weight 0.145689547
slot 0 k3 expert 200 rank 6 local 8  weight 1.34319687
slot 0 k4 expert 53  rank 1 local 21 weight 2.10904183e-08
slot 0 k5 expert 35  rank 1 local 3  weight 0.0106075406
```

Rank `7` local experts `30` and `21` are the immediate non-finite path.
Rank `6` local experts `30` and `8` include the largest route weight but
produce zero gate/down output. This points at expert binding/table/scale
handling or a route layout mismatch, not bad router logits.

The binding trace then showed non-null gated/down weight and scale pointers
with expected strides for all selected experts:

```text
gated weight stride 131072, gated scale stride 4096
down weight stride 65536, down scale stride 4096
```

That reduces the likelihood of a missing pointer-table entry. The diagnostic
bridge's `ffn_normed` route input reaches `max_abs=38.53125`, so the current
highest-probability explanation is that the bridge HC-current path is not yet
producing the activation distribution the MXFP4 experts expect. The next
durable fix is to replace the shared/FFN bridge with the real DS4 FFN
sequence, then re-enable normalized routed expert input under the same trace.

The true shared-FFN gate added `ffn_gate_shexp`, `ffn_up_shexp`, SwiGLU, and
`ffn_down_shexp` execution behind `DS4_V100_TP_EP_TRUE_SHARED_FFN=1`.
The first implementation used FP16 for the SwiGLU midpoint so it could feed
the existing cuBLAS dense path. That was numerically unsafe: layers `0-2`
completed, but layer `3` produced non-finite midpoint/down tensors and the
HTTP request failed with `tp_ep_decode_failed`.

Clamping the FP16 midpoint made the request complete, but it saturated from
layer `2` onward and returned the wrong token:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "MMMMMMMMMMMMMMMM",
  "generated_token_sequence": [36151],
  "decode_tok_s": 50.988668
}
```

The follow-up FP32-midpoint path keeps the shared SwiGLU midpoint in FP32 and
feeds `ffn_down_shexp` through the packed-FP8 scalar kernel. This removes the
half overflow and completes one-token serving, but still fails parity:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": " دايره",
  "generated_token_sequence": [83483],
  "decode_tok_s": 51.259672
}
```

Tensor stats from that run show finite but very large shared-FFN values after
the early layers:

```text
layer 2 shared_mid max_abs=1124516 shared_down max_abs=192612.953
layer 3 shared_mid max_abs=1149645.38 shared_down max_abs=403483.375
layer 4 shared_mid max_abs=5357139.5 shared_down max_abs=683936
```

This does not yet prove the shared FFN itself is wrong, because the serving
harness still carries a partial/proxy hidden-state bridge across layers. It
does prove that a naive FP16 shared-FFN midpoint is not a valid V100
implementation. A production HMMA version needs dynamic scaling or a fused
gate/up/down schedule that does not saturate the midpoint.

Re-enabling `DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1` together with the
FP32-midpoint shared FFN still fails inside the routed executor at layer `0`:

```text
route_input layer 0 rank 7 finite_bad=0 max_abs=38.53125
route_gated layer 0 rank 7 finite_bad=1 max_abs=46176
route_down  layer 0 rank 7 finite_bad=4096 max_abs=1375
```

So the current split is clear:

- true shared FFN is now wired and numerically stable when routed experts stay
  on the old raw-HC input bridge;
- normalized routed expert input remains blocked inside the TurboMind routed
  executor or its input scaling/layout;
- full parity still requires replacing the remaining partial hidden/attention
  bridge with the true DS4 HC attention/FFN sequence.

The routed-normalized nonfinite was fixed by restoring the reference routed
SwiGLU clamp. DS4 clamps routed gate/up with `swiglu_clamp_exp=10`; the
TurboMind gated-SiLU epilogue does not. The TP/EP bridge now uses a clamped
path when `DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1`:

1. plain TurboMind MXFP4 gate/up GEMM into `[gate, up]`;
2. CUDA clamp+SwiGLU into the midpoint;
3. existing TurboMind MXFP4 down GEMM.

The previous layer-`0` HTTP 500 is gone. The fixed run completes all layers
with `decode_finite_bad=0`, but still fails token parity:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "numara",
  "generated_token_sequence": [108877],
  "decode_tok_s": 51.420866
}
```

The route stats now show the intended clamp behavior:

```text
layer 0 rank 7 route_gate_up finite_bad=0 max_abs=413
layer 0 rank 7 route_gated  finite_bad=0 max_abs=100
layer 0 rank 7 route_down   finite_bad=0 max_abs=245.5
```

This closes the immediate routed-executor numeric failure. The next parity
gap is no longer a route crash; it is the remaining proxy hidden/attention/HC
bridge in the TP/EP serving graph.

The compose stage also had one remaining diagnostic shortcut: it added a
synthetic residual instead of the true current hidden shard. The TP/EP compose
kernels now accept `d_current_shard` and fall back to the old synthetic
residual only when no current shard exists. With `DS4_V100_TP_EP_HC_CURRENT_INPUT=1`
the 32-slot / 256K parity run completes without non-finites, but still misses
top-token parity:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "ombonana",
  "generated_token_sequence": [109614],
  "decode_tok_s": 49.703666
}
```

Finally, the true shared-FFN FP32 midpoint was updated to match the reference
DS4 SwiGLU clamp behavior (`gate <= 10`, `up in [-10, 10]`). This removes the
million-scale shared midpoint/down values while preserving FP32 midpoint
materialization:

```text
before shared clamp:
  layer 3 shared_mid max_abs=1148163.88
  layer 3 shared_down max_abs=300071.812

after shared clamp:
  layer 3 shared_mid max_abs=99.9954605
  layer 3 shared_down max_abs=232.418732
```

The clamped-current run still misses parity:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "uerak",
  "generated_token_sequence": [114220],
  "decode_tok_s": 50.094226
}
```

This is useful progress because the observed failure has moved from numerical
instability to a graph-semantics mismatch. Current conclusion: true shared FFN,
routed normalized input, route weights, current residual compose, and endpoint
wiring are live; the remaining parity blocker is the simplified attention/HC
bridge and token-state semantics, not another obvious FFN nonfinite.

The next HC experiment replaced the diagnostic HC bridge with closer reference
semantics: 20 Sinkhorn iterations and no artificial `0.125` scale/clamp in the
HC weighted sum. That exposed the V100-specific activation boundary immediately.
Without a dedicated activation scaling strategy, the routed FFN path overflows
when f32 HC-derived activations are packed into FP16 for TurboMind:

```text
reference HC reduce only:
  route_input layer 2 finite_bad=2000 max_abs=65504

stable RMS + saturating f32->fp16:
  route_input layer 2 finite_bad=0 max_abs=65504
  route_gate_up layer 2 finite_bad=2998 max_abs=65504
  route_down layer 2 finite_bad=8192
  result: HTTP 500
```

So `DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1` is now an explicit diagnostic gate,
not the serving default. The restored default clamped HC bridge still completes
the parity request:

```json
{
  "case": "short_reasoning_plain",
  "expected_text": "16",
  "actual_text": "antsay",
  "generated_token_sequence": [57648],
  "decode_tok_s": 51.298722
}
```

Interpretation: full reference HC semantics are blocked on a real V100
activation quantization/scaling boundary. Simply removing the clamp is not
enough, because the downstream FP16/TurboMind routed executor then overflows.

## Artifacts

- `logs/from-cluster/sprint308-all-local-experts-parity/cluster/server.out`
- `logs/from-cluster/sprint308-all-local-experts-parity/cluster/server.err`
- `logs/from-cluster/sprint308-all-local-experts-parity/cluster/startup.env`
- `logs/from-cluster/sprint308-all-local-experts-parity/cluster/parity-summary.json`
- `logs/from-cluster/sprint308-all-local-experts-parity/cluster/parity.stdout`
- `logs/from-cluster/sprint308-all-local-experts-parity/cluster/parity.stderr`
- `logs/from-cluster/sprint308-route-capacity-parity/cluster/server.out`
- `logs/from-cluster/sprint308-route-capacity-parity/cluster/server.err`
- `logs/from-cluster/sprint308-route-capacity-parity/cluster/parity-summary.json`
- `logs/from-cluster/sprint308-weighted-route-parity/cluster/server.out`
- `logs/from-cluster/sprint308-weighted-route-parity/cluster/server.err`
- `logs/from-cluster/sprint308-weighted-route-parity/cluster/parity-summary.json`
- `logs/from-cluster/sprint308-model-router-routes-parity/20260524-024936/`
- `logs/from-cluster/sprint308-model-router-ffn-norm-parity/20260524-030019/`
- `logs/from-cluster/sprint308-model-router-norm-router-raw-expert-rerun/20260524-034449/`
- `logs/from-cluster/sprint308-model-router-active-mask-parity/20260524-035442/`
- `logs/from-cluster/sprint308-routed-ffn-norm-input-parity/20260524-040200/`
- `logs/from-cluster/sprint308-routed-ffn-norm-input-stats/20260524-040752/`
- `logs/from-cluster/sprint308-routed-ffn-norm-route-ids/20260524-041250/`
- `logs/from-cluster/sprint308-routed-ffn-binding-trace/20260524-041723/`
- `logs/from-cluster/sprint308-true-shared-ffn-parity/20260524-131638/`
- `logs/from-cluster/sprint308-true-shared-ffn-stats/20260524-132106/`
- `logs/from-cluster/sprint308-true-shared-ffn-clamped/20260524-132604/`
- `logs/from-cluster/sprint308-true-shared-ffn-f32mid/20260524-133149/`
- `logs/from-cluster/sprint308-true-shared-f32mid-routed-norm/20260524-133530/`
- `logs/from-cluster/sprint308-routed-norm-clamped-gate/20260524-134337/`
- `logs/from-cluster/sprint308-routed-norm-clamped-gate-v2/20260524-134718/`
- `logs/from-cluster/sprint308-current-residual-compose/20260524-135408/`
- `logs/from-cluster/sprint308-current-residual-shared-clamp/20260524-135928/`
- `logs/from-cluster/sprint308-hc-reference-split-reduce/20260524-140548/`
- `logs/from-cluster/sprint308-hc-stable-rms-reference-reduce/20260524-141051/`
- `logs/from-cluster/sprint308-hc-stable-rms-saturating-half/20260524-141459/`
- `logs/from-cluster/sprint308-default-after-reference-hc-gate/20260524-142053/`

## Production Gate

This sprint remains open. It is complete only when the TP/EP path either
matches the fixed reference vector set or the remaining mismatch is localized
to one concrete layer-stage boundary with logs good enough to implement the
next fix.

Current next fixes:

- keep the routed-normalized clamped path as the correctness path and later
  replace it with a fused clamped TurboMind/CUTLASS executor;
- keep the true shared-FFN path on FP32 midpoint until a scaled/fused HMMA
  version is validated;
- design an explicit activation scale for the reference-HC to FP16/TurboMind
  routed boundary before making reference HC reduce default;
- replace the remaining attention bridge with the full DS4 attention,
  compressed-KV, indexer, and row-selection sequence.
