# Sprint 125 - Batched Attention Output-A Probe

Date: 2026-05-21

## Objective

Improve the 16-slot/256K served throughput path by batching the DS4 attention
output-A projection across all active slots. Sprints 122-124 showed that small
tail fusions are correct but too small to change practical throughput. This
sprint moves to a larger attention-output boundary that is still narrow enough
to validate safely.

## Context

`execute_attention_output_batch()` already batches the attention input
projections and can optionally batch `attn_output_b` for 16 slots. However,
when `DS4_V100_BATCH_ATTN_OUTPUT_B=1`, the loop still runs output-A per slot
through `grouped_attention_output_a()`. That leaves 16 grouped output-A
launches per layer before the single output-B batch.

Output-A is not a plain `8192 x 4096` batch matmul. It is grouped:

- groups: `8`
- rows/group: `1024`
- cols/group: `4096`
- input heads layout per slot: `[8][4096]`
- output low-rank layout per slot: `[8][1024]`

The batched kernel must select the input slice by `row_group`, not by treating
the full heads row as one contiguous 4096-wide input.

## Implementation Plan

1. Add `DS4_V100_BATCH_ATTN_OUTPUT_A=0` and
   `DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=0` launcher knobs.
2. Add a CUDA API:

   ```text
   ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_batch_f32(...)
   ```

   It accepts contiguous batched grouped activations shaped
   `[n_tokens][groups * cols_per_group]` and writes
   `[n_tokens][groups * rows_per_group]`.
3. Add a fixed DS4 HMMA grouped-batch output-A kernel for
   `groups=8`, `rows_per_group=1024`, `cols_per_group=4096`, and
   `n_tokens=16`.
4. Provide a scalar/row-pair grouped-batch fallback in the same API so the path
   remains testable with the HMMA flag off.
5. In `execute_attention_output_batch()`, when the new output-A flag is enabled:
   - run attention/head generation for each slot;
   - defer output-A until all `heads_batch` rows are ready;
   - run one grouped-batch output-A into `low_batch`;
   - run batched output-B when `DS4_V100_BATCH_ATTN_OUTPUT_B=1`, otherwise run
     output-B per slot.
6. Extend `tests/cuda_f8_hmma_attn_batch_smoke.c` to compare grouped-batch
   output-A fallback vs HMMA.

## Validation

Local/static:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_f8_hmma_attn_batch_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

Cluster correctness:

```text
DS4_V100_BATCH_ATTN_OUTPUT_A=1 \
DS4_V100_BATCH_ATTN_OUTPUT_B=1 \
DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1 \
tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144

DS4_V100_BATCH_ATTN_OUTPUT_A=1 \
DS4_V100_BATCH_ATTN_OUTPUT_B=1 \
DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43
```

Served A/B:

```text
tools/ds4-v100-appliance-soak.sh \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --ctx 262144 --slots 16 --active-microbatch 16 \
  --tokens 16 --requests 16 --warmup-requests 1
```

Run a same-binary control with both new flags off, then a candidate with:

```text
DS4_V100_BATCH_ATTN_OUTPUT_A=1
DS4_V100_BATCH_ATTN_OUTPUT_B=1
DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1
```

## Promotion Bar

- 16/16 token match.
- One 16-request tensor batch and 256 tensor-batched tokens.
- No fallback on the output-A candidate path.
- At least roughly 2% generated tok/s improvement over same-binary control.

## Risks

- HMMA accumulation changes numeric order versus the scalar grouped rows2
  kernel; scheduler correctness and token-match soaks are required.
- Deferring output-A changes intra-layer work ordering. It should preserve the
  current per-step stage overlap, but service A/B is the authority.
- Output-B batching was neutral alone in Sprint 122, so this sprint only makes
  sense if output-A batching removes enough grouped per-slot work.

## Status

Completed. Defaults remain off.

## Implementation

Added a guarded grouped-batch attention output-A path:

- `ds4_gpu_arena_f8_e4m3_b128_matmul_grouped_batch_f32`
- a fixed DS4 grouped output-A HMMA kernel for the `8 x 1024 x 4096` shape at
  16 slots;
- a grouped-batch rows2 fallback;
- `DS4_V100_BATCH_ATTN_OUTPUT_A` launcher/runtime flag;
- `DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH`, exported to CUDA as
  `DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH`;
- scheduler wiring that defers output-A until all slot heads are ready, then
  optionally runs output-B as the existing 16-slot batch.

The focused F8 smoke now compares grouped-batch output-A fallback vs HMMA.

## Results

Static/local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build on `llamacpp-build-8gpu` under `/workspace/ds4`:

```text
CUDA_ARCH=sm_70 make -j80 \
  tests/cuda_f8_hmma_attn_batch_smoke \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

Correctness:

```text
cuda_f8_hmma_attn_batch_smoke: ok
cuda_v100_stage_scheduler_smoke: stage=0 gpu=0 layers=0-5 executed=6 tm_layers=6 token=16 pos=16 slots=16 ... ok
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

Trace confirmation from the focused smoke:

```text
ds4: f8_shape_trace kind=grouped_batch path=rows2 gpu=0 rows=8192 cols=4096 n_tokens=16 groups=8 rows_per_group=1024 cols_per_group=4096 calls=1
ds4: f8_shape_trace kind=grouped_batch path=hmma_ds4_attn_o gpu=0 rows=8192 cols=4096 n_tokens=16 groups=8 rows_per_group=1024 cols_per_group=4096 calls=1
```

Same-binary 16-slot/256K served A/B:

| Candidate | Generated tok/s | Continuation tok/s | Correctness | Decision |
|---|---:|---:|---|---|
| control | `43.503005` | `40.784067` | 16/16 token match | baseline |
| output-A HMMA + output-B batch | `43.245208` | `40.542383` | 16/16 token match | reject |
| output-A HMMA only | `43.265219` | `40.561143` | 16/16 token match | reject |
| output-A rows2 batch only | `43.640921` | `40.913364` | 16/16 token match | keep opt-in |
| output-A rows2 + output-B batch | `43.619996` | `40.893746` | 16/16 token match | keep opt-in |

The control and best candidate both reported one tensor-batched group, 16
tensor-batched requests, and 256 tensor-batched tokens.

Launcher config validation with the new flags passed:

```text
DS4_V100_BATCH_ATTN_OUTPUT_A=1 \
DS4_V100_BATCH_ATTN_OUTPUT_B=1 \
DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1 \
tools/ds4-v100-run-appliance.sh --check
```

## Decision

Do not promote a new default. The grouped-batch rows2 output-A path is correct
and slightly faster than the control, but the improvement is about `0.3%`, well
below the roughly 2% promotion bar. The HMMA grouped output-A kernel is correct
but slower in served A/B.

Keep these diagnostics opt-in:

```text
DS4_V100_BATCH_ATTN_OUTPUT_A=1
DS4_V100_BATCH_ATTN_OUTPUT_B=1
DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1
```

The result is consistent with Sprints 123-124: isolated launch consolidation is
not enough. The next sprint should move to a full routed-expert or persistent
kernel boundary rather than another single projection wrapper.
