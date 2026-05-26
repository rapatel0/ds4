# Sprint 411: True-Attention Output Serving Gate

## Goal

Expose the existing true-attention output plus post-attention FFN-input path
through the normal TP/EP HTTP serving harness and measure it at the target
serving shape.

Sprint 410 promoted HC-current NCCL for the fast serving baseline, but that
baseline still reports `scaffold_sum_pre_ep_attention_output_ms=0.0`. For full
DS4 layer semantics, the serving path must be able to run:

- `attn_output_a -> attn_output_b`
- `post_attn = current + attn_output_b`
- FFN norm/router/shared/routed inputs from `post_attn`

This sprint is a semantic progress gate, not a performance-default promotion.
Generated tokens are expected to differ from the incomplete fast baseline.

## Implementation

- Add launcher env `DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT`.
- Wire it to the existing binary flag
  `--true-ds4-post-attention-ffn-input-gate`.
- Add `tools/ds4-v100-tp-ep-profile.py --post-attention-ffn-input`.
- Add `tools/ds4-v100-tp-ep-true-attn-http-ab.py`, a semantic A/B harness:
  - control: current promoted HC-current NCCL fast path
  - candidate: HC-current NCCL plus post-attention FFN input
  - readiness required for both
  - candidate must show non-zero attention-output and post-attention timers
  - response parity with control is recorded only as divergence evidence, not a
    failure condition

## Definition of Done

- [x] Local syntax checks pass for launcher and Python tools.
- [x] V100 target-shape HTTP A/B runs at `32` requests / `32` slots / `256K`
      / `32` generated tokens.
- [ ] Candidate passes readiness with resident KV, typed KV, compact MoE,
      checksums, token-match metadata, GPU samples, and VRAM admission.
- [x] Candidate summary shows non-zero `scaffold_sum_pre_ep_attention_output_ms`
      and `scaffold_sum_pre_ep_post_attention_ffn_input_ms`.
- [x] Sprint, status, vision, and temporary report are updated with measured
      metrics and the next semantic blocker.

## Decision Rule

Mark the semantic path operational if the target-shape candidate passes
readiness and exercises the true-attention/post-attention timers. Do not promote
it as the performance default unless it also preserves reference parity or
explicitly becomes the accepted quality baseline.

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-true-attn-http-ab.py
bash -n tools/ds4-v100-run-appliance.sh
```

V100 target-shape A/B artifact:

```text
logs/from-cluster/sprint411-true-attn-http-ab-rerun/
```

Shape:

```text
32 concurrent HTTP requests
32 configured slots
262144 context
position 262080
32 generated tokens/request
HC-current NCCL enabled
lazy output head enabled
compact MoE decode enabled
model-router routes enabled
```

The first target rerun exposed an incompatibility between post-attention FFN
input and async route-plan upload: `upload_model_router_route_plan_async()`
returns `8` when `routed_ffn_norm_input_gate` is active. The launcher and A/B
harness now force route-plan async upload off for this semantic candidate.

After that fix, the target candidate served all requests and exercised the new
semantic path, but did not pass production readiness because it crossed the
`1536 MiB` NCCL reserve threshold.

| Metric | Control | Post-attn candidate |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| readiness | `true` | `false` |
| server generated decode tok/s | `108.084959` | `20.315962` |
| server continuation decode tok/s | `107.964491` | `20.308358` |
| client generated tok/s | `17.810479266520147` | `8.69354826447401` |
| avg sampled GPU util | `4.198033707865169%` | `7.124305555555556%` |
| max sampled GPU util | `46%` | `24%` |
| min free VRAM | `2106 MiB` | `1328 MiB` |
| VRAM failures | `0` | `62` |
| attention output timer | `0.0 ms` | `512.62943 ms` |
| post-attn FFN input timer | `0.0 ms` | `144.063057 ms` |
| attention projection timer | `54.322579 ms` | `58.002707 ms` |
| attention state timer | `41.194539 ms` | `44.987935 ms` |
| compressed KV timer | `72.464854 ms` | `80.341195 ms` |

## Outcome

Decision:
`true-attention-post-attention-serving-served-reserve-blocked`.

This is a semantic success but not a production promotion. The true-attention
output and post-attention FFN-input path now runs through HTTP serving at the
target shape and produces complete responses. The next blocker is admission and
speed of the real attention-output projection path:

- `true_attention_output_projection` is the new dominant measured semantic
  cost, about `11.5-11.7 ms/layer-step` in representative candidate rows.
- `post_attention_ffn_input` adds about `2.5 ms/layer-step`.
- The path is about `5.3x` slower in server decode than the promoted fast
  baseline and leaves only `1328 MiB` minimum free VRAM against the current
  `1536 MiB` NCCL reserve.

Next work should keep PP/layer-split frozen and continue only on TP/EP:

1. Make post-attention serving memory-admitted at `32` slots / `256K` by
   reducing the attention-output/post-attention scratch and log/stat overhead,
   or by proving a revised NCCL reserve policy with explicit headroom.
2. Replace the current attention-output projection/gather path with the
   intended TP collective/kernel shape before optimizing MoE epilogues.
3. Keep the semantic path default-off until readiness passes and a quality
   baseline is accepted.
