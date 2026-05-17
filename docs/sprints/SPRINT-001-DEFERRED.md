# SPRINT-001 Deferred Items

Items considered during planning but excluded from the current Sprint 001
scope. Sprint 001 is limited to source inventory, architecture reconciliation,
static memory planning, and the pack/runtime contract anchored to
`docs/architecture/DS4-V100-LAYOUT.md`.

## 1. Full Decode Integration

**What:** Run a full DS4 Flash decode through the 8-GPU V100 appliance path.

**Why deferred:** We first need exact source tensor inventory, dtype/dimension
confirmation, and a no-overfill planner. Starting decode before that risks
implementing against the wrong layout.

**Target sprint:** Sprint 002 or later, depending on Sprint 001 verdict.

**Prerequisites:** `SHIP` or bounded `EXTEND` from Sprint 001.

**Files:** `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`, tests.

## 2. Per-Device CUDA State And HC Relay

**What:** Refactor CUDA state into per-visible-device state and implement
hidden-context relay across layer-stage boundaries.

**Why deferred:** The planner should define ownership, memory budgets, and
stage boundaries before CUDA state is split.

**Target sprint:** First implementation sprint after planner acceptance.

**Prerequisites:** Deterministic layer map and per-GPU memory report.

**Files:** `ds4_cuda.cu`, `ds4_gpu.h`, `ds4.c`.

## 3. Loader, Packer, And Runtime Shards

**What:** Add source GGUF type support, offline pack conversion, and per-GPU
runtime shard loading.

**Why deferred:** Sprint 001 defines the exact source-format and manifest
contract. Implementation should follow the confirmed inventory rather than
guessing from expected dtypes.

**Target sprint:** Sprint 002 if inventory matches the architecture baseline.

**Prerequisites:** Model SHA, tensor inventory, source dtype table, and manifest
schema from Sprint 001.

**Files:** `ds4.c`, `gguf` helpers, packer tools, `ds4_cuda.cu`.

## 4. Broad TurboMind Or tc-grid Kernel Import

**What:** Import the V100 TurboMind/tc-grid kernel experiments into DS4.

**Why deferred:** Those kernels are important implementation candidates, but
Sprint 001 only records which tensor families they should target. A broad
kernel import before the pack contract is fixed would create churn.

**Target sprint:** After loader/packer and first execution path are chosen.

**Prerequisites:** Tensor-family runtime layouts and memory budgets.

**Files:** `ds4_cuda.cu`, optional isolated CUDA kernel files.

## 5. INT8 As A Default Runtime Layout

**What:** Convert broad tensor families into INT8 runtime packs by default.

**Why deferred:** INT8 may simplify V100 kernels, but it can expand MXFP4
expert memory and must pass scale-policy and decode-quality gates. Sprint 001
marks INT8 candidate areas only.

**Target sprint:** Future kernel/quality sprint.

**Prerequisites:** Per-family calibration policy, reference comparisons, and
planner proof that INT8-expanded packs still fit.

**Files:** packer tools, `ds4_cuda.cu`, tests.

## 6. Runtime F8 KV Cache Mode

**What:** Add F8/E4M3 KV cache as a selectable runtime mode.

**Why deferred:** F16 KV is the conservative first correctness path. Sprint 001
only budgets F16/F8/F32 KV envelopes.

**Target sprint:** Future long-context/performance sprint.

**Prerequisites:** Baseline decode correctness and measured F16 KV behavior.

**Files:** `ds4.c`, `ds4_cuda.cu`, planner/CLI.

## 7. MTP And Speculative Decoding

**What:** Add native MTP/speculative decode support in the V100 appliance path.

**Why deferred:** MTP is expected upside, but it should not be mixed into the
first topology and format feasibility work.

**Target sprint:** Future performance sprint after base decode.

**Prerequisites:** Coherent non-speculative decode.

**Files:** `ds4.c`, `ds4_server.c`, MTP loading/execution code.

## 8. Server Concurrency And Multi-Slot Scheduling

**What:** Add request batching, slot scheduling, or continuous batching.

**Why deferred:** Sprint 001 only admits slots in the planner. Runtime
scheduling should wait until single-path decode is stable.

**Target sprint:** Future appliance hardening sprint.

**Prerequisites:** Base decode plus slot/context admission report.

**Files:** `ds4_server.c`, `ds4.c`, scheduler code.

## 9. Tensor Parallelism And LM-Head Split

**What:** Implement vocab-parallel output head, routed/shared FFN tensor
parallelism, or full 2-way TP pipeline stages.

**Why deferred:** The architecture document captures these as evaluation
paths. The first implementation should preserve simple layer ownership until
the planner proves where memory or latency pressure actually lands.

**Target sprint:** Future conditional performance sprint.

**Prerequisites:** Baseline layer-sharded planner and at least one decode path.

**Files:** `ds4_cuda.cu`, `ds4.c`, NCCL/build support if needed.

## 10. SSD Or Host-Backed Weight Offload

**What:** Use fast SSD, host-mapped weights, or managed memory as the default
residency strategy.

**Why deferred:** Pure device residency is a core appliance constraint.
Host/SSD paths may be useful for diagnostics only.

**Target sprint:** Future diagnostic/perf comparison only.

**Prerequisites:** Pure VRAM path measured first.

**Files:** loader, packer, CUDA memory management.

## 11. q2/q4 Appliance Fallback

**What:** Use upstream DS4's q2/q4 GGUF family as the primary appliance target.

**Why deferred:** The current target is the high-intelligence quantized DSv4
Flash source layout. q2/q4 remains a fallback/reference option only.

**Target sprint:** Future fallback if the target source layout is infeasible.

**Prerequisites:** Sprint 001 `STOP` caused by source-format or memory fit.

**Files:** `ds4.c`, `download_model.sh`, README.

## 12. VISION.md / Multi-Sprint Roadmap

**What:** Create a longer-term roadmap for the private DS4 appliance fork.

**Why deferred:** The user explicitly deferred roadmap work until feasibility
is understood.

**Target sprint:** After Sprint 001 verdict.

**Prerequisites:** Source inventory and planner verdict.

**Files:** `docs/sprints/VISION.md`.

## Summary

| Item | Target | Required First |
|---|---|---|
| Full decode | Sprint 002+ | Sprint 001 planner verdict |
| Per-device CUDA + HC relay | Sprint 002 | Layer map and memory report |
| Loader/packer/shards | Sprint 002 | Exact inventory and manifest contract |
| TurboMind/tc-grid import | Future implementation | Chosen tensor-family runtime layouts |
| INT8 default | Future quality/perf | Calibration and memory proof |
| F8 KV | Future long-context | F16 decode baseline |
| MTP | Future performance | Non-speculative decode |
| Multi-slot scheduling | Future hardening | Base decode and admission policy |
| TP / LM-head split | Future conditional | Baseline layer-sharded measurements |
| SSD/host offload | Diagnostic only | Pure VRAM path first |
| q2/q4 fallback | Conditional fallback | Target source infeasible |
| VISION.md | After feasibility | Sprint 001 verdict |
