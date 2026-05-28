# TEMP Status Report 417: TP/EP Persistent Graph + NCCL Execution

Date: 2026-05-26

## Scope

Executed the next TP/EP-only performance pass. No PP/layer-split work.

Focus:

- Persistent CUDA graph replay A/B.
- NCCL current-HC allgather admission and performance.
- Slot scaling at 256K context.
- Avoid giving up on an optimization when the bottleneck moved.

## Code Changes

- Added `--tp-runtime-scratch-mib N` to make TP runtime scratch configurable instead of hard-coding 1536 MiB/GPU.
- Added `--defer-nccl-init-gate` so NCCL can be initialized after model residency rather than before expert allocation.
- Kept `--decode-cudagraph-persistent-replay-gate` as the graph replay path.

These are systemic fixes:

- Scratch configurability lets the memory planner reclaim VRAM for communication/runtime variants.
- Deferred NCCL avoids allocating NCCL buffers before the largest resident model allocations.

## Build

Built on gpu-01:

```text
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

## Experiment Results

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint416-exec-matrix/
```

### 8 slots, eager, 256K, 8 decode steps

```text
artifact: eager-slot8-tokens8
PASS
total_decode_ms=1701.322430
aggregate_generated_tok_s_decode=37.617796
aggregate_continuation_tok_s_decode=38.745135
```

Major eager breakdown:

```text
sum_hc_current_input_ms=1358.926858
sum_pre_ep_attention_projection_ms=270.971891
sum_pre_ep_compressed_kv_ms=547.444627
sum_pre_ep_attention_state_ms=142.078208
```

### 8 slots, persistent graph, 256K, 8 decode steps

```text
artifact: persistent-slot8-tokens8
PASS
capture_attempted=43
replay_attempted=344
replay_succeeded=344
total_decode_ms=750.533630
aggregate_generated_tok_s_decode=85.272661
aggregate_continuation_tok_s_decode=89.425145
```

Persistent graph is a 2.27x decode improvement over eager for this test.

### 8 slots, NCCL current-HC without memory fixes

```text
artifact: persistent-nccl-hc-slot8-tokens8
FAIL
error=OOM during expert allocation
```

Even with `--tp-runtime-scratch-mib 512` and unused comp-state disabled:

```text
artifact: persistent-nccl-hc-slot8-tokens8-scratch512
FAIL
error=OOM during expert allocation
```

Slot 4 also failed before deferred NCCL:

```text
artifact: persistent-nccl-hc-slot4-tokens8-scratch512
FAIL
error=OOM during expert allocation
```

Interpretation: the issue was not slot activation size alone. NCCL initialization before expert residency was consuming/fragmenting VRAM too early.

### 8 slots, deferred NCCL current-HC, 256K, 8 decode steps

```text
artifact: persistent-nccl-hc-slot8-tokens8-defer-scratch512
PASS
tp_runtime_scratch=512 MiB/GPU
comp_state=disabled
deferred_nccl=enabled
hc_current_nccl_allgather=enabled
total_decode_ms=714.033120
aggregate_generated_tok_s_decode=89.631697
aggregate_continuation_tok_s_decode=94.646497
```

Improvement over persistent graph without NCCL:

```text
generated decode:      85.27 -> 89.63 tok/s  (+5.1%)
continuation decode:   89.43 -> 94.65 tok/s  (+5.8%)
```

### 16 slots, deferred NCCL current-HC, 256K, 8 decode steps

```text
artifact: persistent-nccl-hc-slot16-tokens8-defer-scratch512
PASS
generated_tokens=128
continuation_tokens=112
total_decode_ms=1095.398430
aggregate_generated_tok_s_decode=116.852459
aggregate_continuation_tok_s_decode=121.222428
```

This is the current best validated throughput point.

### 32 slots, deferred NCCL current-HC, 256K

```text
artifact: persistent-nccl-hc-slot32-tokens8-defer-scratch512
FAIL
kv_bytes_per_gpu=3707940864
scratch_bytes_per_gpu=536870912
error=OOM during expert allocation
```

Interpretation: 32 slots at 256K does not fit with the current all-resident expert + dense F16 cache + KV plan. This needs a memory layout change, not just a smaller scratch buffer.

## Profiling Notes

`nvprof` with CUDA profiler window produced incomplete graph detail; it mostly saw device-7 runtime helper kernels and DtoH/memset events:

```text
bf16_dense_kernel: 1.757 ms
CUDA memcpy DtoH: 1.492 ms total
CUDA memset: 0.618 ms total
dense_kv_slice_kernel: 0.438 ms total
init_hidden_kernel: 0.364 ms total
```

Nsight Compute full all-layer profiling OOMed because profiler overhead pushes the resident model over VRAM. Single-layer NCU needs a separate shared-HC harness; the single-layer path lacks the same shared controls as token-major serving.

## Current Best Metric

```text
Best validated 256K decode throughput:
  slots=16
  persistent graph replay
  deferred NCCL current-HC allgather
  scratch=512 MiB/GPU
  comp_state=disabled
  aggregate_generated_tok_s_decode=116.852459
  aggregate_continuation_tok_s_decode=121.222428
```

## Next Bottleneck

The next material bottleneck is memory layout and residency at higher slots.

Slot 32 probably needs one or more of:

- Remove or compress the all-layer dense F16 cache.
- Convert more dense cached residency to lower precision.
- Stream or lazily materialize some expert shards instead of all-resident active expert buffers.
- Reduce output-head residency during benchmarks unless output quality/top-k measurement is required.
- Add a formal memory planner mode for TP serving that includes NCCL workspace and allocation order.

For performance at the current fit point, the next runtime optimization should be:

- Keep slot 16 as the working benchmark.
- Make deferred NCCL + scratch sizing first-class in the profile harness and serving env.
- Profile a serving-equivalent single-layer/shared-HC path or add lightweight graph node timing counters, because full NCU is too memory-heavy.
