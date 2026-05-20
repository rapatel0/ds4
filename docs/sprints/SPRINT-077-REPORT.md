# Sprint 077 Report: Batched Output-Head Selection

## Outcome

`SHIP_OPT_IN_ONLY`.

Sprint 077 added row-batched output-head projection and row-batched top-1
selection across active slots. The implementation is correct on V100, but the
paired 1M/4-slot throughput run showed it is slower than the Sprint 076 per-slot
parallel device top-1 default. The appliance default remains the per-slot device
top-1 path; batched output-head selection is available for further experiments
with `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1`.

## Changes

- Added `ds4_gpu_arena_bf16_matmul_f32_rows` for row-major batched BF16 output
  projection.
- Added `ds4_gpu_top1_f32_rows_tensor` with deterministic lower-token tie
  handling and non-finite rejection.
- Added scheduler output-head batch scratch and
  `ds4_v100_stage_scheduler_select_token_batch`.
- Updated replay batch generation to select all active slots through one batch
  call when the opt-in is enabled.
- Extended `cuda_v100_bounded_logits_smoke` to cover row matmul/top-1 parity and
  non-finite row rejection.
- Added deployment docs for `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=0`.

## Validation

Local:

```bash
make ds4_v100_scheduler.o ds4_v100_replay.o tools/ds4-v100-replay.o tests/cuda_v100_bounded_logits_smoke.o
git diff --check
```

V100 build:

```bash
CUDA_ARCH=sm_70 make ds4_cuda.o ds4_v100_scheduler.o ds4_v100_replay.o \
  tools/ds4-v100-replay tests/cuda_v100_bounded_logits_smoke \
  tests/cuda_v100_selected_token_smoke
```

V100 smokes:

```bash
./tests/cuda_v100_bounded_logits_smoke
./tests/cuda_v100_selected_token_smoke --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --expected-token-hex 3136 --top-k 1
DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1 ./tests/cuda_v100_selected_token_smoke ...
```

Both selected-token runs selected token `926`, whose bytes match expected hex
`3136`.

## Throughput Evidence

Fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- pack index: `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`
- context: `1048576`
- slots: `4`
- tokens/request: `16`
- measured requests: `4`
- warmup requests: `1`
- async pipeline mode: `per-step`
- expected first token hex: `3136`

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Output-head ms | Avg GPU util | Token match |
|---|---:|---:|---:|---:|---:|---:|
| `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1` | `8.616841` | `8.078288` | `7425.525` | `139.750` | `18.269%` | `4/4` |
| `DS4_V100_DISABLE_OUTPUT_HEAD_BATCH=1` | `9.028544` | `8.464260` | `7086.929` | `135.080` | `19.855%` | `4/4` |
| Default after patch | `9.011829` | `8.448590` | `7099.939` | `135.402` | `19.823%` | `4/4` |

The batched path regressed generated/continuation tok/s by `4.56%`, increased
output-head timing by `3.46%`, and increased latency by `4.78%` versus the
paired per-slot control.

## Decision

Keep the batched output-head primitive as opt-in only. The result indicates that
copying per-slot HC into contiguous scratch plus the row-batched projection does
not beat the already-fast per-slot parallel top-1 path at 4 active slots.

The next throughput work should move back to larger costs: stream/event stage
handoff, kernel-side scheduling, or routed MoE occupancy, rather than more
output-head batching.

## Artifacts

- `logs/from-cluster/sprint077-output-batch`
- `logs/from-cluster/sprint077-output-slot`
- `logs/from-cluster/sprint077-output-default`
- `logs/from-cluster/sprint077-output-batch-optin-smoke`
