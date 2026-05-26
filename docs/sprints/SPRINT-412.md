# Sprint 412: Attention-Output NCCL Semantic Serving Probe

## Goal

Test whether the existing true-attention output NCCL allgather path improves
the Sprint 411 semantic-serving candidate at the real target shape.

Sprint 411 proved that post-attention serving can run end-to-end through HTTP,
but it is not production-admitted and is much slower than the promoted fast
baseline:

- `32/32` HTTP responses served
- `scaffold_sum_pre_ep_attention_output_ms=512.629430`
- `scaffold_sum_pre_ep_post_attention_ffn_input_ms=144.063057`
- server generated decode `20.315962` tok/s versus `108.084959` control
- minimum free VRAM `1328 MiB` versus the `1536 MiB` NCCL reserve

The attention-output projection is therefore the next measured semantic
bottleneck. This sprint checks the already-implemented
`--true-ds4-attention-output-nccl-allgather-gate` under the full post-attention
serving path before adding new kernels.

## Experiment

Run:

```text
tools/ds4-v100-tp-ep-true-attn-http-ab.py
  --ctx 262144
  --slots 32
  --position 262080
  --tokens 32
  --requests 32
  --candidate-attention-output-nccl
```

Control:

- promoted HC-current NCCL fast path

Candidate:

- HC-current NCCL
- true-attention output
- post-attention FFN input
- route-plan async upload disabled
- attention-output NCCL allgather enabled

## Definition of Done

- [x] Local syntax checks pass.
- [x] V100 target-shape HTTP A/B completes or fails with a concrete first
      blocker.
- [x] Record whether attention-output NCCL changes:
      - HTTP success count
      - readiness
      - server decode tok/s
      - attention-output timer
      - post-attention timer
      - minimum free VRAM and reserve failures
- [x] Update sprint/status/vision and commit kept artifacts.

## Decision Rule

Promote nothing unless the candidate is readiness-clean at `32` slots / `256K`
and improves or materially de-risks the Sprint 411 semantic path.

If attention-output NCCL still fails reserve or does not reduce the
attention-output timer, keep it diagnostic-only and move to implementation:
reduce the attention-output projection scratch/log overhead or replace the
projection/gather structure with a purpose-built TP kernel path.

## Implementation

Fixed a profile parser edge case found during the first run. The server can
terminate with a truncated final `tp_ep_token_major_scaffold` line; before this
sprint, that malformed final line overwrote valid earlier scaffold metrics with
`null`. The parser now ignores fields that are absent from truncated scaffold
rows instead of erasing prior values.

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-true-attn-http-ab.py
bash -n tools/ds4-v100-run-appliance.sh
```

V100 target-shape A/B artifact:

```text
logs/from-cluster/sprint412-attn-output-nccl-http-ab-rerun/
```

Shape:

```text
32 concurrent HTTP requests
32 configured slots
262144 context
position 262080
32 generated tokens/request
HC-current NCCL enabled
attention-output NCCL enabled on candidate
lazy output head enabled
compact MoE decode enabled
model-router routes enabled
```

| Metric | Control | Attention-output NCCL candidate |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| readiness | `true` | `false` |
| server generated decode tok/s | `101.539977` | `20.984393` |
| server continuation decode tok/s | `101.187734` | `20.949901` |
| client generated tok/s | `15.744130209984968` | `8.338640536946373` |
| avg sampled GPU util | `2.8587962962962963%` | `6.316287878787879%` |
| max sampled GPU util | `46%` | `25%` |
| min free VRAM | `2106 MiB` | `1328 MiB` |
| VRAM failures | `0` | `62` |
| attention output timer | `0.0 ms` | `486.473759 ms` |
| post-attn FFN input timer | `0.0 ms` | `138.337609 ms` |
| attention projection timer | `54.397643 ms` | `54.613979 ms` |
| attention state timer | `42.933655 ms` | `42.536043 ms` |
| compressed KV timer | `71.699447 ms` | `54.112785 ms` |

For comparison, Sprint 411's non-NCCL post-attention candidate measured:

```text
server generated decode tok/s: 20.315962
attention output timer:        512.629430 ms
post-attn FFN input timer:     144.063057 ms
min free VRAM:                 1328 MiB
VRAM failures:                 62
```

## Outcome

Decision:
`true-attention-post-attention-serving-served-reserve-blocked`.

Attention-output NCCL is a small semantic-path improvement versus Sprint 411,
but not a production unlock:

- server decode improved from `20.315962` to `20.984393` tok/s
- attention-output timer improved from `512.629430` to `486.473759 ms`
- post-attention timer improved from `144.063057` to `138.337609 ms`
- VRAM admission did not change: `1328 MiB` min free and `62` reserve failures

Keep attention-output NCCL diagnostic-only. The next TP/EP implementation work
should target the real semantic path directly:

1. reduce attention-output/post-attention scratch and temporary residency enough
   to pass the `1536 MiB` reserve at `32` slots / `256K`; and
2. replace the attention-output projection/gather sequence with a purpose-built
   TP kernel/collective shape instead of layering another narrow switch.
