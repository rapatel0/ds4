# TEMP Status Report 007

Date: 2026-05-22

## Current Top Line

- Trusted persistent-pack served baseline: 16-slot / 256K reached
  `48.163685` generated tok/s and `47.411127` continuation tok/s with `16/16`
  token matches in Sprint 181.
- Current filled-context direct synthetic baseline after Sprint 190 attention
  scratch default:
  - len-256 / ctx-262144: `14.024529` prompt tok/s,
    `14.659513` continuation tok/s, IDs `3955, 361`.
  - len-1024 / ctx-262144: `14.425868` prompt tok/s,
    `14.429801` continuation tok/s, IDs `926, 926`.
- Historical served runs before the persistent-pack refresh reached roughly
  `70-71` generated tok/s at 16-slot / 256K, but the current comparable
  trusted baseline is the Sprint 181 persistent-pack number above.

## Sprint 191 Progress

Shipped an opt-in attention-detail profiler:

- `DS4_V100_PROFILE_ATTENTION_DETAIL=1`
- JSON buckets under `timing_ms.stage_profile`:
  - `attn_proj`
  - `attn_cache`
  - `attn_softmax`
  - `attn_inverse_rope`
  - `attn_output`

V100 build passed on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

## Latest Profile Signal

For len-1024 / ctx-262144:

| Bucket | Time ms | Share of total | Share of attention |
|---|---:|---:|---:|
| Attention total | 37195.555 | 56.25% | 100.00% |
| Attention output | 15478.988 | 23.41% | 41.62% |
| Q/KV projection | 11803.892 | 17.85% | 31.73% |
| Softmax | 5429.091 | 8.21% | 14.60% |
| Cache/update | 3739.496 | 5.66% | 10.05% |
| Inverse RoPE | 383.503 | 0.58% | 1.03% |
| FFN | 21656.052 | 32.75% | n/a |

Conclusion: standalone RoPE and inter-stage transfer are not the next material
lever. The next practical target is the attention output/projection boundary.

## Techniques Explored So Far

- TurboMind MXFP4 routed expert kernels copied/adapted into this repo.
- Fused TurboMind gate/up pack path, now default for the production pack.
- Fixed-route and down-reduce six-route routed executors: correct but flat or
  slower in served A/B.
- Software-pipelined TurboMind gate/up variants: isolated improvement, not a
  served-path win.
- Wider slot admissions: 16-slot/256K, 32-slot/128K, 64-slot/64K,
  128-slot/32K, 256-slot/16K.
- TP/EP probes: 2-way primitive positive in isolation, but current per-layer
  copy-back overlay regressed served throughput.
- Online single-token attention: promising served speedup but output diverged
  after several tokens, so it remains default-off.
- Attention scratch reuse: promoted as default-on for attention-only
  single-slot path after matching-output V100 validation.

## Remaining Work

- Implement a real attention projection/output optimization, likely one of:
  - persistent or larger fused grouped attention-output path,
  - projection/output fusion with low-precision source bytes expanded inside
    the GPU,
  - TP/EP shape change that densifies F8 projection/output work without
    per-layer full-hidden copy-back.
- Re-run served 16-slot / 256K aggregate throughput after the attention change.
- Revisit MTP only after base decode throughput moves; current MTP verify works
  diagnostically but true draft commit is not a shipped speedup.
- Decide whether persistent TP/EP ownership is worth a larger topology sprint
  if attention fusion does not move the served baseline.

## Evidence

- `logs/from-cluster/sprint191-attn-detail/len256-detail-fixed/result.json`
- `logs/from-cluster/sprint191-attn-detail/len256-detail-fixed/summary.json`
- `logs/from-cluster/sprint191-attn-detail/len1024-detail/result.json`
- `logs/from-cluster/sprint191-attn-detail/len1024-detail/summary.json`
