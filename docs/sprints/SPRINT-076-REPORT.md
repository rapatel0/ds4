# Sprint 076 Report: Parallel Output-Head Top-1 Reducer

## Summary

Sprint 076 replaced the Sprint 075 serial device top-1 scan with a parallel
two-stage CUDA reducer. The path is correct, detects non-finite logits, and is
now the default greedy `k == 1` output-head selector because V100 evidence
showed a material throughput and output-head timing win.

## Implementation

- Replaced the one-thread `top1_f32_serial_kernel` with:
  - `top1_f32_blocks_kernel`, which scans vocabulary chunks in parallel;
  - `top1_f32_final_kernel`, which reduces chunk winners to one token/logit.
- Preserved deterministic tie handling:
  - larger logit wins;
  - exact equal logits keep the lower token id.
- Added raw exponent-bit non-finite detection so the fast path fails closed if
  any logit is NaN or infinity.
- Kept `ds4_gpu_top1_f32_tensor` API stable.
- Switched greedy `k == 1` output selection to default-on after the A/B result.
- Kept rollback controls:
  - `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1`;
  - `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=0`.
- Preserved the full-logit host top-k path for `k > 1` and diagnostics.

## V100 Validation

Build:

```bash
CUDA_ARCH=sm_70 make ds4_cuda.o \
  tests/cuda_v100_bounded_logits_smoke \
  tests/cuda_v100_selected_token_smoke \
  tools/ds4-v100-replay
```

Correctness:

```text
cuda_v100_bounded_logits_smoke: ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
```

Selected-token smoke passed in three modes:

- host-logit control before the default flip;
- explicit parallel top-1 with `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`;
- default-on parallel top-1 after the scheduler policy flip.

The disable fallback also passed:

```bash
DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1 ./tests/cuda_v100_selected_token_smoke ...
```

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

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Output-head ms | Stage 7 decode ms | Avg GPU util | Max GPU util | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| host-logit control | `8.656498` | `8.115467` | `7391.521` | `324.953` | `1979.423` | `19.075%` | `40.000%` | baseline |
| parallel device top-1 | `9.031197` | `8.466747` | `7084.718` | `134.510` | `1470.800` | `19.271%` | `40.000%` | `+4.329%` generated |

Output-head timing dropped by `58.606%`, and average request latency dropped by
`4.151%`.

## Decision

Make the parallel device top-1 path the default for greedy `k == 1` output
selection. It clears both default-change gates:

- generated tok/s improved by more than `1%`;
- output-head timing improved by more than `10%`;
- selected-token correctness stayed green.

Keep the host-logit path as the rollback and diagnostic path through
`DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1` and for `k > 1`.

## Artifacts

- `logs/from-cluster/sprint076-top1-parallel`
- `logs/from-cluster/sprint076-top1-default`
- `logs/from-cluster/sprint076-top1-comparison`

## Validation

- local C/CUDA-facing compile where possible;
- `git diff --check`;
- V100 CUDA build for replay and smokes;
- V100 bounded logits smoke;
- V100 selected-token smoke for control, explicit fast path, default-on path,
  and disable fallback;
- V100 sustained A/B at 1M context, 4 slots, per-step async;
- JSON artifact validation.
