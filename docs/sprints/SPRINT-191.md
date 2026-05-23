# Sprint 191 - Attention Detail Profile Buckets

Date: 2026-05-22

## Objective

Add opt-in attention sub-bucket profiling so filled-context work can distinguish
projection/cache/softmax/output cost instead of treating the whole attention
body as one opaque bucket.

## Rationale

Sprint 190 made attention-only single-slot scratch the default. The next
optimization target should be based on the new baseline. Current
`--profile-decode` reports only the coarse `attention` bucket around
`execute_attention_output()`, which hides whether the remaining cost is
projection, KV/cache update, softmax, inverse RoPE, or output projection.

## Scope

- Add default-off `DS4_V100_PROFILE_ATTENTION_DETAIL=1`.
- Populate attention-detail timing fields in the layer, stage, replay counters,
  and JSON output.
- Keep normal runtime unchanged unless profiling is enabled.
- Validate on V100 with direct synthetic replay and `--profile-decode`.

## Non-Goals

- No kernel optimization in this sprint.
- No online-attention promotion.
- No TP/EP topology work.
- No default behavior change beyond Sprint 190's already-shipped attention
  scratch default.

## Definition of Done

- [x] V100 build passes.
- [x] Direct synthetic profile emits non-zero attention detail buckets.
- [x] Evidence is archived under `logs/from-cluster/`.
- [x] Sprint outcome identifies the next target bucket.
- [x] Vision is updated.
- [x] Changes are committed.

## Implementation

- Added `DS4_V100_PROFILE_ATTENTION_DETAIL=1` as a default-off diagnostic
  profile mode.
- Split the single-slot attention body into reportable buckets:
  projection, cache/update, softmax, inverse RoPE, and attention output.
- Propagated the new buckets through layer reports, stage scheduler reports,
  replay counters, and JSON `timing_ms.stage_profile`.
- Preserved the attention-detail fields across FFN report population so the
  routed-FFN report fill does not clear the already-measured attention data.

## V100 Evidence

Build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Direct synthetic profile, `ctx=262144`, `tokens=2`,
`DS4_V100_PROFILE_ATTENTION_DETAIL=1`, persistent Sprint 181 pack:

| Prompt len | Prompt tok/s | Continuation tok/s | Output IDs | Attention ms | Projection ms | Cache ms | Softmax ms | Inverse RoPE ms | Output ms |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 256 | 14.024529 | 14.659513 | `3955, 361` | 9021.159 | 2956.831 | 919.591 | 1000.459 | 89.455 | 3871.865 |
| 1024 | 14.425868 | 14.429801 | `926, 926` | 37195.555 | 11803.892 | 3739.496 | 5429.091 | 383.503 | 15478.988 |

For len-1024, total profiled stage time was `66127.631 ms`. Attention was
`56.25%` of total and FFN was `32.75%`. Within attention, output projection was
the largest sub-bucket (`41.62%` of attention), followed by q/kv projection
(`31.73%`), softmax (`14.60%`), cache/update (`10.05%`), and inverse RoPE
(`1.03%`).

Evidence:

- `logs/from-cluster/sprint191-attn-detail/len256-detail-fixed/result.json`
- `logs/from-cluster/sprint191-attn-detail/len256-detail-fixed/summary.json`
- `logs/from-cluster/sprint191-attn-detail/len1024-detail/result.json`
- `logs/from-cluster/sprint191-attn-detail/len1024-detail/summary.json`

## Outcome

Sprint 191 shipped the missing visibility layer. The filled-context profile
does not point at standalone RoPE or handoff as the next lever. The next
implementation should target the attention output/projection boundary:

- persistent or larger fused grouped attention-output path,
- projection/output fusion that keeps low-precision source bytes resident and
  expands inside the GPU,
- or a TP/EP boundary that makes these F8 projection shapes denser without
  adding full-hidden copy-back per layer.

## Follow-Up

Sprint 192 should implement one of the attention projection/output levers and
validate against the len-256 and len-1024 synthetic tiers before moving back to
served 16-slot/256K aggregate throughput.
