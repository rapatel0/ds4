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

## Production Gate

This sprint remains open. It is complete only when the TP/EP path either
matches the fixed reference vector set or the remaining mismatch is localized
to one concrete layer-stage boundary with logs good enough to implement the
next fix.

Current next fixes:

- add an isolated TurboMind/route microbench for `ffn_normed` routed expert
  input, starting with the layer-`0` failure reproduced by
  `DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1`;
- implement the true shared-FFN gate/up/SwiGLU/down subpath instead of the
  current bridge;
- replace the remaining attention bridge with the full DS4 attention,
  compressed-KV, indexer, and row-selection sequence.
