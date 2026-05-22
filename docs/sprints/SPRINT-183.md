# Sprint 183 - Single-Token Online Attention Decode Gate

Date: 2026-05-22

## Objective

Test the most direct attention/KV lever from Sprint 182: route single-token
long-context decode through the existing heads8 online attention kernel instead
of the score-buffer attention kernel.

## Rationale

Sprint 182 changed the bottleneck picture for the 256K practical-serving tier.
The synchronized profile showed attention/KV work dominating visible stage
time, and the async profile showed large stage/host wait. The code already has
`attention_decode_mixed_heads8_online_kernel`, but the normal decode path only
uses it when the score buffer is too small, or for larger token batches under
the window-attention gate. Single-token serving at 256K still usually uses
`attention_decode_mixed_kernel`.

The online kernel avoids score-buffer materialization and computes softmax
online over raw plus compressed rows. It is already used elsewhere, so this
sprint adds a narrow opt-in selector before considering a larger attention
rewrite.

## Scope

- Add an explicit opt-in env gate for online single-token decode.
- Apply it to both arena-backed attention decode and batch attention decode.
- Keep defaults unchanged until V100 evidence proves a production win.
- Validate correctness on the persistent Sprint 181 appliance pack.
- Benchmark the 16-slot / 256K production-pack path against the current
  default.

## Non-Goals

- No new attention kernel implementation.
- No change to default serving behavior unless the A/B clears the gate.
- No TP/EP topology rewrite in this sprint.
- No MTP promotion.

## Implementation Notes

Add a helper such as:

```c
static int cuda_attention_decode_online_single_enabled(void)
```

The selector should require:

- `DS4_CUDA_ATTENTION_DECODE_ONLINE_SINGLE=1`
- `DS4_CUDA_NO_WINDOW_ATTENTION` not set
- no compressed attention mask
- `head_dim == 512`
- `n_tokens == 1`

The existing online kernel supports `n_tokens == 1`, `ratio == 0`, and
ratio-compressed decode. It should therefore be selectable in:

- `ds4_gpu_arena_attention_decode_heads_tensor`
- `attention_decode_batch_launch`

## Definition of Done

- [x] The opt-in env gate compiles locally.
- [x] V100 build passes for `tools/ds4-v100-replay`.
- [x] Selected-token smoke passes with the gate enabled.
- [x] 16-slot / 256K sustained A/B is run against the persistent production
      appliance pack.
- [x] Decision is recorded: promote if it improves production serving without
      correctness loss; otherwise keep default-off.
- [x] Sprint 183 evidence is copied into `logs/from-cluster/`.
- [x] Vision is updated.
- [x] Changes are committed.

## Outcome

Implemented `DS4_CUDA_ATTENTION_DECODE_ONLINE_SINGLE=1`, a default-off CUDA
attention selector that routes single-token decode through the existing
`attention_decode_mixed_heads8_online_kernel` when:

- `DS4_CUDA_NO_WINDOW_ATTENTION` is not set
- no compressed attention mask is active
- `head_dim == 512`
- `n_tokens == 1`

The selector applies to:

- `ds4_gpu_attention_decode_heads_tensor`
- `ds4_gpu_arena_attention_decode_heads_tensor`
- `attention_decode_batch_launch`

The first smoke exposed a real semantic mismatch in the existing online kernel:
`ratio == 0` / `n_tokens == 1` already made all compressed rows visible, but
did not mirror the baseline kernel's raw-window behavior. The sprint fixed that
by making the online kernel also keep up to `256` raw rows visible in this
single-token all-visible mode.

## Evidence

Build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed on `llm/llamacpp-build-8gpu`.

Selected-token smoke with the gate enabled passed after the raw-window fix:

| Context | Slots | Tokens/request | Requests | Generated tok/s | Continuation tok/s | Match |
|---:|---:|---:|---:|---:|---:|---:|
| 262144 | 1 | 2 | 1 | `0.752578` | `0.376289` | 1/1 |

The full 16-slot / 256K same-binary A/B was positive:

| Mode | Context | Slots | Tokens/request | Requests | Generated tok/s | Continuation tok/s | Match | Avg GPU util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 262144 | 16 | 64 | 16 | `47.648307` | `46.903802` | 16/16 | `30.295%` |
| online-single | 262144 | 16 | 64 | 16 | `49.378552` | `48.607012` | 16/16 | `30.877%` |

That is a `+3.63%` generated-token gain and a `+3.63%` continuation-token gain
for the measured production-pack run.

Direct 8-token JSON comparison:

| Step | Control token | Online-single token | Match |
|---:|---|---|---|
| 0 | `926` / `16` | `926` / `16` | yes |
| 1 | `1` / `<end>` | `1` / `<end>` | yes |
| 2 | `380` / ` P` | `380` / ` P` | yes |
| 3 | `223` / space | `223` / space | yes |
| 4 | `11154` / `204` | `11154` / `204` | yes |
| 5 | `26` / `8` | `26` / `8` | yes |
| 6 | `1492` / ` /` | `17` / `/` | no |
| 7 | `223` / space | `7833` / `128` | no |

The divergence starts after small floating-point differences accumulate into a
different greedy top-1 token. This is not automatically a quality failure, but
it is not bit-identical enough to promote as a production default without a
broader output-quality gate.

Cluster evidence:

```text
logs/from-cluster/sprint183-online-attn/
```

## Decision

Keep `DS4_CUDA_ATTENTION_DECODE_ONLINE_SINGLE` default-off for now.

This is the first attention/KV-path change in the current sequence that shows a
clear sustained 16-slot / 256K throughput improvement. However, it changes
multi-token greedy output. Because the project is optimizing for source-model
quality, the right next step is not blind promotion; it is either:

1. Run a broader deterministic quality/tolerance gate for the online attention
   path and promote if acceptable, or
2. Use the result as evidence that persistent/online attention is a real lever
   and implement a quality-preserving version of the same idea.

The sprint therefore moves the high-throughput serving vision forward, but does
not claim the practical-serving target is realized.
