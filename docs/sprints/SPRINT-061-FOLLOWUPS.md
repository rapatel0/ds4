# Sprint 061 Follow-ups

## Slot Scaling Is Flat

- **Severity**: Critical
- **Target sprint**: Sprint 062
- **Files**: `ds4_v100_scheduler.c`, `ds4_v100_replay.c`,
  `tools/ds4-v100-sustained-decode-bench.sh`
- **Issue**: Four active slots at 256K context do not improve aggregate
  throughput. The measured result is `3.834046` generated tok/s, worse than
  the two-slot 1M Sprint 060 reference `3.915266`.
- **Evidence**: `logs/from-cluster/sprint061-persistent-views/sustained-256k-s4/`.
- **Next step**: Stop assuming more slots will help under the current
  layer-synchronous schedule. Evaluate stage wavefronting, committed MTP, or a
  persistent decode loop that keeps GPUs busy on different token batches.

## Shared F8 Batch Is Correct But Not Default-Fast

- **Severity**: Important
- **Target sprint**: Sprint 062+
- **Files**: `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_v100_layer_execute.c`
- **Issue**: The batched shared F8 source path reduces launches but does not
  beat the existing per-slot shared path on V100.
- **Evidence**: The no-extra-pointer-upload variant measured `3.884237`
  generated tok/s at two slots, below Sprint 060's `3.915266`.
- **Next step**: Keep `DS4_V100_BATCH_SHARED_F8=1` as an experiment. Do not
  make it default unless profiling shows a fix, such as a better F8 tile kernel
  or removal of another synchronization point.

## Kernel-Level Profiling Needed

- **Severity**: Important
- **Target sprint**: Sprint 062
- **Files**: `ds4_cuda.cu`, `tools/ds4-v100-sustained-decode-bench.sh`
- **Issue**: GPU utilization remains around `11-12%` even when the batch path
  is active. Without kernel timing, it is unclear whether the next bottleneck
  is row-reduction arithmetic, launch latency, host scheduling, attention/KV,
  or inter-stage idle time.
- **Evidence**: Sprint 061 sustained runs all remain below `4` generated tok/s
  aggregate with low average utilization.
- **Next step**: Add a low-overhead CUDA event timing mode around attention,
  routed FFN, shared FFN, HC, and output-head sections, then use the timing
  split to choose between MTP commit, stage wavefronting, or kernel rewrite.
