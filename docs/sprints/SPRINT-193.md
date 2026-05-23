# Sprint 193 - Single-Slot F8 F16 Cache cuBLAS Gate

Date: 2026-05-22

## Objective

Move beyond wrapper-level kernel swaps by testing a resident-weight execution
boundary for single-slot F8 attention projection/output matmuls.

## Rationale

Sprint 191 showed filled-context decode is dominated by attention. Sprint 192
proved that reusing the 16-slot grouped HMMA attention-output kernel for
single-token decode is the wrong shape. However, the codebase already has a
more structural mechanism: `DS4_CUDA_F8_F16_CACHE=1` expands F8 arena weights to
resident F16 and uses cuBLAS for batched F8 matmuls. The current implementation
intentionally excludes single-token decode with `n_tokens > 1`.

V100 does not have native FP8/FP4 tensor cores. For practical serving, the next
reasonable experiment is to pay a one-time resident F8-to-F16 weight expansion
for selected attention F8 matrices and use Volta tensor-core/cuBLAS kernels for
single-slot decode, while preserving source quantized packs and keeping a VRAM
reserve gate.

This directly tests whether avoiding per-token F8 dequantization inside the
row-pair scalar kernels can move the measured q/kv projection and output-B
costs without changing model quality.

## Scope

- Add default-off `DS4_CUDA_F8_F16_CACHE_SINGLE=1`.
- Extend `ds4_gpu_arena_f8_e4m3_b128_matmul_f32()` to use the existing
  resident F16 arena cache plus cuBLAS for approved single-token F8 shapes.
- Keep the existing row-pair kernels as default and automatic fallback.
- Wire launcher/env-example controls and VRAM reserve documentation.
- Validate direct synthetic len-256 and len-1024 with
  `DS4_V100_PROFILE_ATTENTION_DETAIL=1`.
- Record continuation tok/s and attention sub-bucket movement.

## Non-Goals

- No default promotion unless V100 A/B is clearly positive and VRAM fit is
  clean.
- No grouped attention-output-A rewrite in this sprint.
- No TP/EP topology change.
- No MTP change.
- No full 256K prefill benchmark.

## Implementation Plan

1. Add `DS4_CUDA_F8_F16_CACHE_SINGLE` parsing in `ds4_cuda.cu`.
2. Add a conservative single-token shape allowlist:
   - `1024 x 4096` q_a-like attention projection
   - `512 x 4096` kv_latent-like attention projection
   - `32768 x 1024` q_b-like attention projection
   - `4096 x 8192` output-B / dense projection
3. In `ds4_gpu_arena_f8_e4m3_b128_matmul_f32()`, before scalar row kernels,
   if the gate is enabled, cuBLAS is ready, and F16 cache allocation succeeds:
   - convert the single F32 activation vector to F16 scratch,
   - call `cublasGemmEx(..., n=1, CUDA_R_16F inputs, CUDA_R_32F accum/output)`,
   - return on success, otherwise fall back to the existing path.
4. Add launcher/env-example controls:
   - `DS4_V100_CUDA_F8_F16_CACHE`
   - `DS4_V100_CUDA_F8_F16_CACHE_SINGLE`
   - `DS4_V100_CUDA_F8_F16_CACHE_RESERVE_MIB`
5. V100 A/B:
   - control
   - cache single candidate with enough reserve
   - len-256 / ctx-262144 / tokens=2
   - len-1024 / ctx-262144 / tokens=2

## Definition of Done

- [x] V100 build passes.
- [x] Candidate correctness gate evaluated; it failed at len-256, so len-1024
  was intentionally skipped.
- [x] Candidate does not overfill VRAM; cache logs or nvidia-smi/free-memory
  evidence are recorded.
- [x] Timing evidence is archived under `logs/from-cluster/`.
- [x] Outcome states whether q/kv projection or output-B moved enough to keep
  pursuing resident F16 cache, or whether Sprint 194 should pivot to
  persistent TP/EP ownership.
- [x] Vision is updated.
- [x] Changes are committed.

## Implementation Attempt

- Temporarily added default-off `DS4_CUDA_F8_F16_CACHE_SINGLE=1`.
- Temporarily added a conservative single-token shape allowlist in
  `ds4_cuda.cu`.
- Temporarily added an opt-in single-token path in
  `ds4_gpu_arena_f8_e4m3_b128_matmul_f32()` that attempted resident F8->F16
  cache plus cuBLAS and fell back to the current row kernels.
- Temporarily wired launcher/env-example knobs:
  - `DS4_V100_CUDA_F8_F16_CACHE`
  - `DS4_V100_CUDA_F8_F16_CACHE_SINGLE`
  - `DS4_V100_CUDA_F8_F16_CACHE_RESERVE_MIB`
- After V100 validation failed the correctness gate, these runtime/code changes
  were removed before commit. The committed artifact is the sprint record and
  evidence, not a retained unsafe runtime flag.

## V100 Evidence

Build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

Direct synthetic len-256 / ctx-262144 / tokens=2:

| Mode | Prompt tok/s | Continuation tok/s | Output IDs | Attention ms | Projection ms | Output ms | Total profile ms |
|---|---:|---:|---|---:|---:|---:|---:|
| control | 13.805395 | 14.575734 | `3955, 361` | 9059.668 | 2957.457 | 3887.584 | 16667.472 |
| F8-F16 cache single candidate | 13.838530 | 14.714705 | `201, 5` | 9155.222 | 2988.009 | 3897.414 | 16755.129 |

The candidate did not overfill VRAM in this bounded run; `nvidia-smi` reported
all devices free before and after the process because allocations were released
on exit. The correctness gate failed before len-1024. A len-8 shape-trace run
showed the intended `cublas_f16_single` path did not select in direct replay;
the trace remained on `rows1` F8 paths for the relevant shapes. That means the
existing resident cache mechanism is not a valid single-slot promotion path
without deeper cuBLAS/device-handle work and a dedicated correctness gate.

Evidence:

- `logs/from-cluster/sprint193-f8-f16-cache-single/len256-control/result.json`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len256-control/summary.json`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len256-cache/result.json`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len256-cache/summary.json`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len256-cache/stderr.log`
- `logs/from-cluster/sprint193-f8-f16-cache-single/before-nvidia-smi.txt`
- `logs/from-cluster/sprint193-f8-f16-cache-single/after-len256-nvidia-smi.txt`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len8-trace/result.json`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len8-trace/summary.json`
- `logs/from-cluster/sprint193-f8-f16-cache-single/len8-trace/stderr.log`

## Outcome

Rejected for production. Even before the intended cuBLAS single path selected,
enabling the resident F8-F16 cache gate changed output IDs on the len-256
filled-context test. That fails the model-quality requirement.

This closes the “easy resident F16 cache” branch for single-slot decode. The
next sprint should pivot to persistent TP/EP ownership or an explicitly
correctness-tested fused attention kernel. It should not rely on the existing
global cuBLAS/F16-cache machinery as a shortcut.
