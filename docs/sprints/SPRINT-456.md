# Sprint 456: Skip Redundant Slot-Major FFN Norm in Rank-Major Serving

## Objective

Stay TP/EP-only and test whether the current router+FFN rank-major serving
baseline can remove the old slot-major FFN norm staging path at the target
serving shape.

The candidate gate is:

```text
DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM=1
```

## Rationale

Sprint 455 promoted the cleanest current `32` slot / `256K` serving baseline:

```text
DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB=1280
DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
```

With router logits and shared/routed FFN input consumers reading rank-major
post-attention state, the slot-major `hc->d_ffn_normed` materialization should
be unnecessary for the optimized serving path unless a diagnostic, stats path,
or unconverted consumer still depends on it.

Earlier Sprint 440 rejected this idea at a smaller graph shape because token
output changed. Since then, the rank-major input parity and router+FFN serving
bundle have changed materially. This sprint retests the candidate against the
current serving baseline, not the old graph proxy.

## Implementation

No new kernel should be required unless validation shows a missing consumer.
Use the existing gate and harness support:

- `--post-attention-skip-slot-major-ffn-norm-gate`
- `DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`

Run same-binary HTTP A/B:

```text
shape:       32 requests / 32 slots / 256K context / 32 generated tokens
scratch:     1280 MiB
baseline:    router+FFN rank-major default bundle
candidate:   baseline + skip slot-major FFN norm
```

## Definition of Done

- Control and candidate both pass readiness.
- Response parity matches `32/32`.
- Candidate preserves the first-token stream pairwise.
- Candidate improves server generated decode tok/s by at least `1.02x`, or the
  sprint records a concrete rejection reason.
- Candidate keeps `vram_failures=0` against the current reserve.
- Results are recorded in a TEMP status report and in `VISION.md`.

## Decision Rule

Promote the gate only if the full target-shape HTTP A/B passes readiness,
response parity, and throughput. If parity fails or throughput is flat/slower,
keep the gate diagnostic-only and move the next sprint back to graph-safe
launch reduction or broader HC/post-attention staging.

## Outcome

Artifact:

```text
/localpool/ds4/workspace/logs/s456-skip-slot-major-ffn-norm-s32-t32
```

The exclusive rerun completed at the full target shape.

| Metric | Control | Candidate | Candidate/control |
|---|---:|---:|---:|
| Readiness | `true` | `true` | pass |
| HTTP 200 | `32` | `32` | pass |
| Response parity | `0/32 matched` | `0/32 matched` | fail |
| Output-head first token | `109865` | `109865` | matched |
| Response-0 first token | `104565` | `104565` | matched |
| Response checksum | `17913667570271397799` | `17913667564178658333` | mismatch |
| Server generated decode tok/s | `34.999820` | `35.421446` | `1.0120x` |
| Server continuation decode tok/s | `35.039950` | `35.392239` | `1.0101x` |
| Client generated tok/s | `14.767353` | `14.791231` | `1.0016x` |
| Average GPU util | `11.845652%` | `11.674145%` | `0.9855x` |
| HC-current input ms | `393.599693` | `391.250157` | `0.9940x` |
| HC-current gather ms | `5.893971` | `5.763197` | `0.9778x` |
| Min free VRAM | `1734 MiB` | `1734 MiB` | `1.0000x` |
| VRAM failures | `0` | `0` | pass |

The candidate removed the slot-major FFN norm gate
(`scaffold_post_attention_skip_slot_major_ffn_norm_gate=1`) and slightly
reduced HC-current/post-attention timing, but it did not meet the throughput
threshold and it failed full response parity. Pairwise first tokens matched,
but response checksums differed for all `32` responses.

## Decision

Do not promote
`DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM=1`.

Keep the gate diagnostic-only. The result confirms that the slot-major FFN norm
materialization is still semantically visible over continuation decode, even
when router logits and routed/shared FFN inputs are rank-major consumers.

## Follow-Up

Move the next sprint to always-on lightweight profiling and short steady-state
heavy-profile windows. The live run showed startup memory waves followed by a
more parallel but low-utilization steady phase; the bottleneck is now more
likely duty cycle, launch/sync fragmentation, or low-intensity memory work than
the isolated slot-major FFN norm copy.
