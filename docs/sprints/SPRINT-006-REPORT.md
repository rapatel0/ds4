# SPRINT-006 Report: Multi-GPU Execution Context And Layer Skeleton

## Verdict

`SHIP`

Sprint 006 produced the sidecar V100 execution context, fail-closed
execution-format policy, descriptor binding, no-math layer skeleton, CUDA
resource ownership, and HC relay smoke requested by the plan. Source-layout
generation remains guarded.

## What Shipped

- `ds4_v100_context.h` / `ds4_v100_context.c`
  - static 8-stage DS4 V100 layer map;
  - V100 source dtype, tensor-family, and execution-kind policy;
  - descriptor binding from pack rows with dtype/shape/byte-length/owner/span
    validation;
  - per-layer family summaries for the skeleton walk;
  - memory reserve accounting fields;
  - report output that states BF16, FP8, and FP4 are not native V100 tensor-core
    formats.
- `ds4_v100_context_cuda.cu`
  - CUDA device fact collection;
  - per-stage stream, relay stream, cuBLAS handle, scratch, and relay-buffer
    ownership;
  - production topology checks using 8 visible V100s, compute capability 7.0,
    32 GB-class VRAM, and peer access;
  - FP16 normal relay and FP32 debug relay smoke path without host-backed relay
    fallback.
- `tools/ds4-v100-context-smoke`
  - opens the context from the current pack index;
  - emits policy, memory, descriptor, and layer-skeleton reports;
  - walks all 43 layers without launching decode math.
- Tests:
  - `tests/v100_context_smoke`
  - `tests/cuda_v100_context_smoke`
  - `tests/cuda_hc_relay_smoke`

## V100 Precision Policy

The runtime policy now explicitly rejects the wrong interpretation of the
source model:

- BF16 is source/probe/explicit-conversion only.
- FP8 and MXFP4/FP4 are packed source/runtime inputs to later registered
  kernels.
- Production dense GEMMs on V100 target FP16 HMMA with FP32 accumulation.
- FP32 is for control/reduction/debug paths, not a broad default GEMM fallback.
- There is still no claim of native BF16, FP8, or FP4 tensor-core execution on
  V100.

## Validation

Local validation:

```text
make tools/ds4-v100-residency-smoke tools/ds4-v100-context-smoke \
  tests/v100_context_smoke tests/pack_index_smoke tests/gpu_arena_smoke \
  tests/bf16_probe_smoke
./tests/pack_index_smoke
./tests/gpu_arena_smoke
./tests/bf16_probe_smoke
./tests/v100_context_smoke
tools/ds4-v100-context-smoke --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --slots 4 --scratch-bytes 1048576 --reserve-mib 2048 \
  --planned-kv-mib 1024 --output-head-mib 1024 --mtp-mib 512 \
  --f32-debug-relay
tests/residency_smoke_synthetic.sh
git diff --check
```

Cluster validation on `llamacpp-build-8gpu`:

```text
make cpu tests/pack_index_smoke tests/gpu_arena_smoke tests/bf16_probe_smoke \
  tests/v100_context_smoke tools/ds4-v100-context-smoke CUDA_ARCH=sm_70
make tests/cuda_v100_context_smoke tests/cuda_hc_relay_smoke CUDA_ARCH=sm_70
./tests/cuda_v100_context_smoke --production \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --planned-kv-mib 1024 --reserve-mib 2048 \
  --output-head-mib 1024 --mtp-mib 512
./tests/cuda_hc_relay_smoke
```

Archived artifacts:

- `docs/sprints/drafts/SPRINT-006-LOCAL-SMOKE.log`
- `docs/sprints/drafts/SPRINT-006-CONTEXT-SMOKE.log`
- `docs/sprints/drafts/SPRINT-006-CUDA-CONTEXT.log`
- `docs/sprints/drafts/SPRINT-006-RELAY.log`
- `docs/sprints/drafts/SPRINT-006-GUARD.log`

Key cluster facts:

- 8 visible Tesla V100-SXM2-32GB devices.
- All visible devices report compute capability 7.0.
- Peer matrix is fully connected in the validation pod.
- Real pack context binds 1328 descriptors.
- Execution-kind counts:
  - F32 control: 687
  - F16 HMMA after conversion: 365
  - low-bit kernel: 129
  - diagnostic only: 147
  - unsupported: 0
- Layer skeleton validates all 43 layers.
- HC relay smoke passes.
- Source-layout generation guard still exits with code 1 and the expected
  source-layout rejection message.

## Deviations

- The context smoke tool is host-side and does not itself run the CUDA relay.
  CUDA relay is validated by `tests/cuda_hc_relay_smoke`, and the production
  CUDA topology/memory check is validated by `tests/cuda_v100_context_smoke`.
  This keeps the tool usable on macOS and keeps CUDA linkage isolated.
- UUID matching is not mandatory yet. The CUDA context logs PCI bus IDs and the
  peer matrix; UUID enforcement remains an open production-mode hardening item.

## Remaining Scope

All deferred items remain deferred in `docs/sprints/SPRINT-006-DEFERRED.md`:
decode, prefill, KV population, real FP8/MXFP4/INT kernels, output-head math,
MTP, tensor-parallel exceptions, server deployment, and throughput tuning.
