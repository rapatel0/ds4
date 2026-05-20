# Sprint 075 Report: Output-Head Top-1 Fast Path

## Summary

Sprint 075 added an output-head top-1 CUDA primitive and persistent output-head
scratch, then tested it against the existing host-logit path on the 8x V100
cluster. The path is correct, but it is not a throughput win in its current
serial-reducer form, so it is committed as an opt-in diagnostic path instead of
the default serving path.

## Implementation

- Added `ds4_gpu_top1_f32_tensor` for F32 logits.
- Added output-head scratch tensors to `ds4_v100_stage_scheduler`.
- Reused the scratch for HC norm, output-head collapse, output embedding,
  output norm, and full-vocab logits.
- Added an opt-in `k == 1` selection path behind
  `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`.
- Kept the previous full-logit host top-k path as the default and for `k > 1`.
- Added `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1` as an override when both
  env vars are present.
- Added a bounded logits smoke check for the top-1 CUDA primitive.

## V100 Validation

Build:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-replay \
  tests/cuda_v100_selected_token_smoke \
  tests/cuda_v100_bounded_logits_smoke \
  tests/cuda_v100_output_head_parity_smoke
```

Correctness:

```text
cuda_v100_bounded_logits_smoke: ok
cuda_v100_output_head_parity_smoke: cpu_top1=83253 gpu_top1=83253 ... ok
cuda_v100_selected_token_smoke: selected=926 ... expected=3136 ... ok
```

The selected-token smoke also passed with the host-logit fallback path.

## Throughput Matrix

Fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- context: `1048576`
- slots: `4`
- queue policy: `sequential`
- async pipeline mode: `per-step`
- tokens/request: `16`
- measured requests/case: `4`
- warmup requests/case: `1`
- expected first token hex: `3136`

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Output-head ms | Avg GPU util | Max GPU util | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| host-logit default | `8.659254` | `8.118051` | `7389.251` | `346.461` | `19.285%` | `40.000%` | baseline |
| device top-1 candidate | `8.697510` | `8.153916` | `7356.500` | `423.818` | `18.800%` | `40.000%` | `+0.442%` generated |

The candidate showed a tiny aggregate throughput increase, but output-head
timing regressed by `+22.328%`. That means the current one-thread CUDA scan is
slower than copying `129280` logits to the host and scanning there.

## Decision

Do not make this the default. Keep the existing host-logit output-head path as
the practical serving path and leave the device top-1 path opt-in for future
parallel reducer work.

The next useful output-head sprint should either:

- implement a real parallel deterministic reducer; or
- batch output-head projection/top-1 across active slots.

Otherwise, practical throughput work should return to larger decode costs:
stream/event handoff, removing stage-level synchronizes, or kernel-side MoE
execution work.

## Artifacts

- `logs/from-cluster/sprint075-output-fast`
- `logs/from-cluster/sprint075-output-fallback`
- `logs/from-cluster/sprint075-output-comparison`

## Validation

- local C/CUDA-facing object compile;
- V100 CUDA build for replay and smokes;
- V100 bounded logits smoke;
- V100 output-head parity smoke;
- V100 selected-token smoke with candidate and fallback paths;
- V100 sustained A/B at 1M context, 4 slots, per-step async;
- JSON artifact validation;
- `git diff --check`.
