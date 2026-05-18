# Sprint 006 Intent: Multi-GPU Execution Context And Layer Skeleton

## Seed

Plan the next sprint after Sprint 005. The goal from `docs/sprints/VISION.md`
is to introduce the production 8-GPU execution context and a layer-owned
execution skeleton with hidden-context relay boundaries and explicit V100
execution-format policy.

This sprint must incorporate the Sprint 005 correction: V100 does not support
native BF16, FP8, or FP4 tensor-core execution. BF16 source bytes can be
gathered and expanded exactly for diagnostics, but production GEMMs must target
V100-native FP16 tensor-core math or the selected low-bit/integer kernel
families after offline packing.

## Context

- Sprints 001-004 proved source-model inventory, manifesting, pack indexing,
  shard emission, and full 8x V100 device residency for all 1328 tensors.
- Sprint 005 shipped the first resident source-dtype diagnostic:
  `token_embd.weight` BF16 bytes can be read from device arena memory and
  expanded to F32 bit-exactly on V100. Normal source-model generation remains
  guarded.
- The next useful milestone is not more diagnostics in isolation; it is a
  production-shaped multi-GPU execution context that can own per-GPU weight
  arenas, scratch, streams, handles, relay buffers, and tensor descriptors.
- `docs/architecture/DS4-V100-LAYOUT.md` is the topology and format anchor:
  baseline is 8 contiguous layer shards, HC-only boundary transfers, FP16
  activations, F16 KV first, source FP8/MXFP4 packs feeding V100 kernels, and
  no persistent dequantized weight copies.
- Existing CUDA code is still mostly global/single-device (`g_cublas`,
  `g_cuda_tmp`, model-cache globals). Sprint 006 should avoid a broad decode
  rewrite and instead create a sidecar production context plus a minimal layer
  skeleton contract.

## Recent Sprint Context

- `8814a83 Ship DS4 V100 BF16 gather probe`: CUDA and host-stub BF16
  row-gather/expand, real GGUF and shard provider probe logs, guard log, and
  Sprint 005 report.
- `975aee9 Add BF16 arena probe stub`: model-less host reference and exact
  BF16 bit-pattern tests.
- `13cbdef Complete DS4 V100 pack residency sprint`: full pack residency,
  GGUF/shard provider smoke, and V100 memory evidence.

## Vision Context

The North Star is a narrow DeepSeek V4 Flash appliance for the 8x
V100-SXM2-32GB cluster that runs the high-intelligence source quantized model
from pure device-resident packs by default.

Sprint 006 sits between the resident tensor probe and source-model decode
correctness. It should make the runtime shape credible without enabling normal
source generation. The guard should remain active until a later sprint proves
correct decode against a reference.

Parking-lot items now relevant:

- Production multi-GPU execution context.
- HC relay and layer scheduler.
- Device-resident output and stream-aware variants of the Sprint 005 probe.
- Source-layout embedding dtype fix or replacement path.
- Additional BF16 tensor probes only if they materially validate the skeleton.

Parking-lot items still deferred:

- Full source-model decode correctness.
- Prefill, compressed KV, indexer integration, and long-context slot admission.
- FP8/MXFP4 routed expert kernel implementation.
- MTP/speculative decoding.
- Tensor-parallel exceptions.
- Server deployment.

## Relevant Codebase Areas

- `ds4_gpu.h`: current public CUDA/Metal tensor API plus residency-only arena
  API. Candidate location for opaque `ds4_gpu_context` and descriptor structs.
- `ds4_cuda.cu`: global CUDA state, tensor allocation, arena implementation,
  BF16 diagnostic gather, cuBLAS handle, temp scratch, existing CUDA kernels.
- `ds4.c`: engine open/guard, pack reconciliation, source-layout detection,
  Metal graph-style scheduler functions, embedding calls, decode/prefill
  orchestration, MTP code that must remain out of scope.
- `ds4_pack.c` / `ds4_pack.h`: pack index lookup and per-GPU byte accounting.
- `tools/ds4-v100-residency-smoke.c`: proven pack upload/probe harness, useful
  for context smoke tests but not the production runtime itself.
- `docs/architecture/DS4-V100-LAYOUT.md`: layer map, dtype/layout table,
  memory budget, tensor-parallel alternatives, and context assumptions.

## Constraints

- Preserve the source-model generation guard unless an explicit validation path
  proves correctness. Sprint 006 should not claim runnable decode.
- Keep pure device residency. Do not introduce managed-memory, host-backed, or
  SSD-backed successful paths.
- Do not materialize persistent dequantized FP16/F32 copies of large source
  weights. Runtime descriptors may reference packed source/V100 pack bytes.
- Do not treat BF16/FP8/FP4 as native V100 compute formats. Execution policy
  must be explicit: FP32 for small control/reduction/debug, FP16 for tensor-core
  activations/GEMMs, low-bit/integer kernels where validated.
- Keep edits narrow. Avoid rewriting the existing Metal graph scheduler or
  CUDA legacy model-cache path.
- The first runtime skeleton should be layer-owned and contiguous across 8
  GPUs. Tensor parallelism remains a planned exception, not the default.
- The V100 pod is disposable; cluster validation must copy logs back into the
  repo.

## Success Criteria

- A production-shaped `ds4_gpu_context` or equivalent opaque context exists for
  8 GPUs, with per-GPU stream/handle/scratch/weight-arena/relay ownership.
- The context can be initialized from a pack index and report the baseline
  layer map, per-GPU arena sizes, memory kind, reserve/headroom, and P2P matrix.
- The context has typed resident tensor descriptors for at least global BF16
  embedding and a representative layer-owned tensor span, without relying on
  GGUF offsets at execution call sites.
- The execution-format policy is encoded in docs and emitted by a smoke tool or
  diagnostic report: BF16 gather/expand diagnostic, FP16 activation target,
  FP32 control/reduction target, FP8/MXFP4 source packs feeding future kernels.
- A minimal HC relay primitive or relay-buffer smoke exists for boundary
  payload shape `[active_slots][4][4096]`, proving device-to-device or fallback
  transfer semantics without running decode.
- A layer skeleton can walk the planned 8-stage layer map and validate tensor
  ownership, descriptor presence, and boundary relay allocations without
  executing attention/MoE math.
- Local tests pass, CUDA synthetic context tests pass on V100, real pack smoke
  reports are archived, and `git diff --check` passes.

## Verification Strategy

- Model-less tests: validate planner/layer map, descriptor bounds, execution
  policy classification, and invalid config failures.
- CUDA synthetic tests: initialize an 8-GPU-capable context on the V100 pod,
  allocate small per-GPU arenas, create relay buffers, and perform a small HC
  relay transfer when at least two GPUs are visible.
- Real pack diagnostic: use `/models/DSv4-Flash-256e-fixed.gguf` plus
  `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` to initialize context metadata
  and allocate bounded/probe-only arenas if full residency is not required.
- Guard validation: normal source-model generation must still fail closed.
- Reports: archive context topology/memory/policy logs under
  `docs/sprints/drafts/`.

## Uncertainty Assessment

- Correctness uncertainty: Medium. This sprint can verify ownership, pointer,
  relay, and policy mechanics, but not numerical layer correctness.
- Scope uncertainty: Medium. A full production context could sprawl into decode;
  the plan should define a skeleton that stops before attention/MoE execution.
- Architecture uncertainty: High. Existing CUDA state is global and legacy
  model-cache oriented, while the appliance needs per-GPU ownership. The sprint
  must choose a context shape that can coexist with existing code without a
  destabilizing rewrite.

## Open Questions

1. Should Sprint 006 allocate full real per-GPU weight arenas again, or should
   it use metadata/probe-only arenas and rely on Sprint 004 for full residency?
2. Should the first HC relay primitive require CUDA peer access, or should it
   implement a pinned-host fallback in the same sprint?
3. How much of `ds4.c` should know about the new context in Sprint 006:
   engine-open wiring only, or a standalone diagnostic tool first?
4. Should typed descriptors live in `ds4_gpu.h`, `ds4_pack.h`, or a new
   `ds4_v100_context.h` to avoid overloading the residency-only arena API?
5. What is the minimal layer skeleton output: a context report only, or a
   no-math walk that validates every tensor family required by one layer?
