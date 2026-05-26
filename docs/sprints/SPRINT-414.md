# Sprint 414: Semantic Attention/Post-Attention Skip-Stats Gate

## Goal

Reduce measured overhead in the TP/EP semantic serving path by removing
diagnostic tensor-stat collection from the true attention-output projection and
post-attention FFN-input stages when running production-style HTTP serving.

Sprint 413 made the semantic path operational at reduced slots and selected
`28` slots / `256K` as the practical benchmark tier. The candidate remains
about `5x` slower than the fast control, with active timers concentrated in:

- `scaffold_sum_pre_ep_attention_output_ms=422.604024`
- `scaffold_sum_pre_ep_post_attention_ffn_input_ms=128.822428`
- `scaffold_sum_ep_ms=450.805272`
- `scaffold_sum_hc_current_input_ms=847.411614`

The attention-output and post-attention functions still collect max/finite
stats inside the timed region for every layer. Earlier compressed-KV work
proved stats collection can create meaningful host/device synchronization
overhead. This sprint makes those stats optional and tests whether the
production semantic path improves at the practical `28` slot tier.

## Implementation

Add a default-off gate:

```text
DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=1
--true-ds4-semantic-skip-stats-gate
```

When enabled:

- skip `collect_tensor_f32_stats` in `run_true_ds4_attention_output_projection`
- skip `collect_tensor_f32_stats` in `run_true_ds4_post_attention_ffn_input`
- still preserve kernel execution, NCCL allgather, router route planning, and
  HTTP serving behavior
- print `stats_skipped=1` in the relevant per-layer rows so artifacts are
  auditable

Expose the gate through:

- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-true-attn-http-ab.py`

## Validation

Local:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py \
  tools/ds4-v100-tp-ep-true-attn-http-ab.py
```

Cluster:

```text
tools/ds4-v100-tp-ep-true-attn-http-ab.py
  --ctx 262144
  --slots 28
  --position 262080
  --tokens 32
  --requests 28
  --candidate-attention-output-nccl
  --candidate-semantic-skip-stats
```

## Definition of Done

- [x] Local syntax checks pass.
- [x] V100 build succeeds.
- [x] `28` slot / `256K` HTTP A/B completes.
- [x] Candidate remains readiness-clean with semantic timers active.
- [x] Record server decode, client generated throughput, VRAM, GPU util, and
      attention-output/post-attention timers.
- [x] Promote the gate only if readiness remains clean and server decode or
      semantic timers improve materially.
- [x] Update docs/status/vision and commit kept artifacts.

## Decision Rule

Promote the skip-stats gate as a semantic-serving default only if it preserves
HTTP readiness and materially improves the `28` slot semantic path.

If it is flat, keep it diagnostic-only and move to the next real kernel lever:
attention-output projection/allgather fusion or post-attention full-hidden
replication removal.

## Outcome

Decision:
`semantic-skip-stats-promoted`.

The skip-stats gate preserves readiness at the practical semantic-serving
shape and materially improves the production-style semantic path. It is now
enabled automatically by the launcher when TP/EP serving runs with
`DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=1`.

The gate only removes diagnostic host-visible tensor stats from timed semantic
sections. It does not skip attention-output kernels, post-attention residual
work, RMS/router inputs, NCCL, HTTP serving, KV, or MoE execution.

Cluster artifacts:

```text
logs/from-cluster/sprint414-semantic-noskip-28slot-http-ab/
logs/from-cluster/sprint414-semantic-skip-stats-28slot-http-ab/
```

Shape:

```text
ctx      = 262144
position = 262080
requests = 28
slots    = 28
tokens   = 32/request
```

| Case | HTTP | Ready | Server decode tok/s | Continuation tok/s | Client generated tok/s | GPU util avg | GPU util max | Min free VRAM | VRAM failures | Attention output ms | Post-attn ms | HC-current ms | EP ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no skip | `28/28` | `true` | `19.708590` | `19.693809` | `7.543522997079557` | `5.844230769230769%` | `24.0%` | `1790 MiB` | `0` | `460.797268` | `129.119597` | `880.779219` | `446.687363` |
| skip stats | `28/28` | `true` | `31.091919` | `31.064390` | `10.366506446092782` | `7.899436090225564%` | `37.0%` | `1790 MiB` | `0` | `19.520681` | `82.034351` | `395.013270` | `444.291619` |

Measured improvement:

- server decode: `1.578x`
- continuation decode: `1.577x`
- client generated throughput: `1.374x`
- attention-output timed section: about `23.6x` lower
- post-attention timed section: about `1.57x` lower
- HC-current timed section: about `2.23x` lower

The no-skip and skip-stats semantic candidates produced the same response-0
token sequence:

```text
[32461, 124727, 73288, 123477, 107880, 63104, 95158, 32974,
 58572, 32974, 33611, 26343, 1853, 123477, 110614, 70623,
 64811, 118187, 3090, 23824, 14868, 39913, 6256, 50615,
 27623, 32461, 43048, 90042, 128818, 117160, 25689, 91569]
```

This removes diagnostic synchronization as a major source of semantic-path
measurement distortion, but it does not make the semantic path fast enough.
The promoted fast control at the same `28` slot shape remains about
`98 tok/s`, while the skip-stats semantic candidate is `31.091919 tok/s`.
The next TP/EP-only implementation target should stay on NCCL/semantic
boundaries: remove the GPU0 full-hidden gather/broadcast in
`run_true_ds4_post_attention_ffn_input` or replace it with a sharded/NCCL
post-attention path before returning to `32` slots.
