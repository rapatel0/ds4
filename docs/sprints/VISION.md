---
created: 2026-05-17
last_updated: 2026-05-23
last_updated_by: vision
revision: 231
archived_previous: docs/sprints/archive/VISION-2026-05-23-pre-tp-hard-cut.md
---

# Vision: DS4 V100 TP/EP Appliance

## North Star

Build a DeepSeek V4 Flash appliance for the 8x V100-SXM2-32GB cluster that
runs the source quantized model from pure device-resident packs, preserves
quality, and reaches practical high-throughput serving through a native
TP/EP topology.

Hard cut: from this revision forward, no new work is spent on PP/layer-split
variants. The old layer-scheduled appliance remains only a frozen correctness
and throughput baseline. All new implementation work targets TP/EP. MTP is
deferred until TP/EP serving is operational and benchmarked.

Target topology:

```text
8x V100:
  pipeline parallel = 1
  tensor parallel   = 8
  expert parallel   = 8
  KV cache          = sharded
  slots target      = 32
  context target    = 256K minimum
  model path        = source quantized, device resident
```

Every GPU should participate in every layer. Dense paths are tensor-parallel.
Routed MoE paths are expert-parallel, using the existing low-bit TurboMind /
CUTLASS kernel work where it helps. The execution goal is to make decode look
like batched mat-mat work over active slots, not single-slot mat-vec work and
not a serial layer-chain.

## Current State

- The PP/layer-scheduled appliance is deployed and useful as a baseline, but
  it is no longer the optimization target.
- Sprint 225 fixed the immediate MTP reset/snapshot blocker:
  `long_memory_archive` full-prompt reset parity and target-block restore now
  pass.
- Sprint 225 also corrected the benchmark contract:
  single-slot replay is diagnostic only, while practical throughput must be
  measured with multi-slot serving and `active_microbatch == slots`.
- The current frozen production-shaped PP baseline from Sprint 225 is:
  `32` slots / `256K`, `64/64` token match, `50.434232` generated tok/s,
  `47.282093` continuation tok/s, average GPU utilization `47.076%`, max
  GPU utilization `96%`.
- Existing TP work is prototype-only. TP is not operational in production
  serving.
- Sprint 226 converted the TP planner into a TP8/EP8-only contract. It no
  longer exposes PP/layer-split topology modes. Against the real production
  pack bytes, the target `32` slots / `256K` / F8-KV shape fits at about
  `27.00 GiB` per GPU including a `2.00 GiB` reserve, with `5.00 GiB`
  headroom.
- Sprint 227 built the TP8 collective workbench. The doubling all-reduce
  boundary is correct and density-sensitive: `1189` overhead-only tok/s at
  32 tokens, `2119` at 64, and `3332` at 128 for the 43-layer,
  two-collective proxy. Root/direct RS+AG is correct but slower and is not the
  first runtime boundary candidate.
- Sprint 228 emitted the TP/EP pack contract from the real production pack.
  The contract has dense TP rows, replicated control/router rows, EP expert
  ownership, and KV/state descriptors, with a balanced `27.024 GiB` per-GPU
  estimate at `32` slots / `256K` / F8 KV.
- Sprint 229 added the first separate TP runtime skeleton. It opens all eight
  GPUs, enables peer access, allocates target hidden/KV/compression/scratch
  arenas for `32` slots / `256K`, runs a fixture pass, and tears down cleanly.
- Sprint 230 added explicit per-layer sharded KV row ownership to the separate
  TP runtime. Ratio-4/indexer and ratio-128 dense/KV slices pass on the V100
  pod at `32` slots / `256K` / F8 KV with `max_abs=0`.
- Sprint 231 added the bounded EP routed-expert slice. A new TP/EP-only smoke
  runs the real TurboMind MXFP4 grouped gated-SiLU and grouped down kernels on
  all eight V100s at the `32` slot / `top_k=6` target, with finite exact repeat
  output and explicit route/latency reporting.
- Prior TP evidence remains useful:
  - TP8 sharded KV at `32` slots / `256K` fits, while replicated KV does not.
  - TP8 one-layer synthetic and FP16 fixture probes proved resident TP work can
    live inside an all-GPU boundary.
  - The current TurboMind MXFP4 TP8 shard-256 path failed correctness; TP4
    controls were correct but did not justify production integration.
  - Routed-only overlays and PP scheduler TP patches are rejected.

## Non-Negotiable Constraints

- No new PP/layer-split optimization sprints.
- No generic scheduler abstraction to support both PP and TP.
- TP/EP code uses separate files and a separate runtime ownership model.
- PP code may be read for reference and used as a frozen baseline, but not
  extended as the forward path.
- Single-slot tests are correctness/latency diagnostics only.
- Throughput evidence must use multi-slot server mode, report prompt tok/s,
  generated tok/s, continuation tok/s, GPU utilization, and confirm
  `active_microbatch == slots`.
- MTP stays out of the critical path until TP/EP serving is correct and
  measured.

## Sprint Sequence

### Sprint 226 - TP/EP Planner And Topology Contract [complete]

Goal: Create a TP-only planner and topology report for `PP1/TP8/EP8` at
`32` slots / `256K`.

Rationale: The PP planner carries legacy assumptions that will fight the new
topology. The TP path needs its own memory, KV, expert, collective, and slot
admission contract before runtime work starts.

Outcome: Complete. `tools/ds4-v100-plan-tp.c` is now a TP8/EP8-only planner
with sharded KV, expert ownership, route-density, admission-tier, and
collective/EP traffic reporting. The real-pack V100 run reports `145.42 GiB`
total resident weight bytes, `27.00 GiB` per-GPU total at `32` slots / `256K`
/ F8 KV, and admission of `63` slots at `256K` under current assumptions.

### Sprint 227 - TP8 Collective Workbench [complete]

Goal: Build TP-only collective smokes for hidden all-reduce, reduce-scatter,
all-gather, and expert-output reduction across all eight V100s.

Rationale: The suspected TP risk is not raw NVLink bandwidth alone; it is
latency, synchronization, and whether collectives can stay resident and
overlapped inside the layer boundary.

Outcome: Complete. `tools/ds4-v100-tp8-collective-workbench` now measures
`allreduce`, `reduce-scatter`, `allgather`, `rs-ag`, and `ep-reduce` modes.
At 32 tokens, the hidden all-reduce proxy is `26.904544 ms` and the EP reduce
proxy is `27.436756 ms`; both pass correctness. At 128 tokens they improve to
`3332.257` and `3253.920` overhead-only tok/s respectively.

### Sprint 228 - TP/EP Pack Contract [complete]

Goal: Emit a TP/EP pack layout with dense TP shards, EP expert ownership, KV
shard descriptors, and per-GPU memory accounting.

Rationale: Runtime work should not reinterpret PP pack metadata. The pack
format must encode the TP/EP ownership model directly.

Outcome: Complete. `tools/ds4-v100-tp-ep-pack-contract` emits
`tp-ep-pack-contract.tsv`, `tp-ep-memory-summary.tsv`, and
`tp-ep-pack-contract.md`. The real-pack contract has `4096` dense TP rows,
`5496` replicated control/router rows, `688` EP expert rows, and `840`
KV/state rows. Per-GPU total is `27.024 GiB` at the target shape.

### Sprint 229 - TP Runtime Skeleton [complete]

Goal: Add a new TP-only runtime skeleton that opens all eight GPUs, allocates
resident hidden/KV/scratch arenas, and executes no-op or fixture layer passes.

Rationale: The runtime must prove ownership, lifecycle, and memory residency
without touching `ds4_v100_scheduler.*` as a shared abstraction.

Outcome: Complete. `ds4_v100_tp_runtime.{h,cu}` and
`tools/ds4-v100-tp-runtime-smoke.cu` now provide a separate TP runtime
skeleton. The V100 smoke allocates `7061329920` runtime bytes per GPU before
weights at the target shape and verifies fixture output with
`fixture_max_abs=0`.

### Sprint 230 - TP Dense And KV Slice [complete]

Goal: Implement a bounded dense-attention/KV slice in the TP runtime, including
sharded DS4 compressed KV at the `32` slot / `256K` target.

Rationale: TP must keep hidden state and KV in native sharded layout across
layers. This sprint answers whether dense paths and KV are viable before MoE
complexity is added.

Outcome: Complete. `ds4_v100_tp_runtime_dense_kv_slice` now computes
per-layer, per-slot sharded KV offsets and writes/reads deterministic resident
KV rows on all eight GPUs. At the target `32` slots / `256K` / F8 KV shape,
the runtime allocates `7122628608` bytes per GPU before weights. Layer 2
ratio-4 with indexer KV passes at `attn_row=384`, `indexer_row=256`,
`attn_row_bytes=65`, `indexer_row_bytes=17`, and `max_abs=0`. Layer 3
ratio-128 without indexer KV passes at `attn_row=192`, `attn_row_bytes=65`,
and `max_abs=0`. This keeps the TP runtime path viable and moves the next
implementation gate to EP routed experts.

### Sprint 231 - EP Routed Expert Slice [complete]

Goal: Implement a bounded EP routed-expert slice using real low-bit expert
kernels and measure expert dispatch, route imbalance, and grouped GEMM density
at `32` active slots.

Rationale: Expert execution dominates the useful work. EP is only valuable if
active slots create dense enough expert batches and dispatch/reduction does not
erase the kernel gains.

Outcome: Complete. `tools/ds4-v100-tp-ep-expert-smoke.cu` models EP8
ownership as `256` global experts and `32` local experts per GPU, then runs
the real TurboMind MXFP4 grouped gated-SiLU and grouped down kernels on all
eight V100s. At `32` slots / `top_k=6`, it reports `192` aggregate routes,
`1.5 MiB` dispatch, `1.5 MiB` return, balanced route imbalance `1.0`,
`repeat_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`, and `PASS`. Rank `7` is
the slow rank at `0.249378 ms` versus roughly `0.059 ms` on ranks `0-6`, so
per-rank timing must remain visible in Sprint 232.

### Sprint 232 - One-Layer TP/EP Correctness Gate [planned]

Goal: Execute one real DS4 layer end to end in the TP/EP runtime and compare
against the frozen PP baseline.

Rationale: This is the first point where dense TP, sharded KV, router, EP
experts, shared path, collectives, and residual state meet.

Outcome: Pending.

### Sprint 233 - Full-Layer TP/EP Decode [planned]

Goal: Scale the TP/EP runtime from one layer to all 43 layers with MTP off and
prove selected-token/decode correctness.

Rationale: TP is not operational until all layers run in the TP/EP ownership
model without gathering hidden state back to PP layout between layers.

Outcome: Pending.

### Sprint 234 - TP/EP Serving Gate [planned]

Goal: Serve continuous multi-slot requests through the TP/EP runtime at
`32` slots / `256K`, MTP off.

Rationale: This is the first true replacement candidate for the frozen PP
baseline. It must report generated tok/s, continuation tok/s, prompt tok/s,
GPU utilization, collective time, expert time, and correctness.

Outcome: Pending.

### Sprint 235 - TP/EP Throughput Optimization [tentative]

Goal: Optimize the measured TP/EP bottleneck after Sprint 234 identifies it.

Rationale: The bottleneck may be collectives, expert dispatch, KV bandwidth,
dense projection layout, output head, or host scheduling. The optimization
sprint should follow measured evidence, not guesswork.

Outcome: Pending.

### Sprint 236 - Multi-Slot MTP On TP/EP [tentative]

Goal: Add MTP to the TP/EP serving path only after TP/EP decode is correct and
benchmarked.

Rationale: MTP should multiply a good decode runtime, not hide a topology
problem. Single-slot MTP diagnostics are not throughput evidence.

Outcome: Pending.

## Experiment Backlog

These experiments should be run inside the TP/EP sprints, not as PP variants:

- TP8 collective roofline at `M=32/64/128`, hidden `4096`.
- TP8 dense GEMM fixture using FP16/FP8-style low-bit expansion on GPU.
- TP sharded KV allocation/update/read at `32` slots / `256K`, then `512K`
  if memory allows.
- EP routed expert smoke with real TurboMind/CUTLASS low-bit kernels at
  `32` active slots.
- Expert load-balance measurement: active experts, routes per expert, and
  worst-GPU imbalance.
- One-layer TP/EP correctness against frozen PP baseline.
- Full 43-layer TP/EP decode correctness.
- TP/EP serving throughput with generated and continuation tok/s separated.

## Parking Lot

- PP/layer-split scheduling optimizations: archived. Use only as baseline.
- Routed-only TP overlays inside the PP scheduler: rejected.
- Generic PP/TP scheduler abstraction: rejected.
- Single-slot throughput reports: rejected as practical-serving evidence.
- MTP serving: deferred until TP/EP serving is operational.
- PP-oriented MTP block-2 promotion: paused; useful correctness evidence only.

## Pivot Log

| Date | Change | Rationale | Next |
|---|---|---|---|
| 2026-05-23 | Archived the prior PP-era vision to `docs/sprints/archive/VISION-2026-05-23-pre-tp-hard-cut.md`. | The accumulated roadmap still documents history, but it no longer reflects the strategy. | Use this file as the active alignment document. |
| 2026-05-23 | Sprint 230 proved TP sharded KV row ownership at `32` slots / `256K`. | TP/EP needs resident hidden/KV state before EP expert work is meaningful. | Build the bounded EP routed-expert slice in separate TP/EP files. |
| 2026-05-23 | Sprint 231 proved bounded EP8 routed expert execution with real TurboMind MXFP4 kernels. | The EP low-bit kernel path is live outside the PP scheduler, but rank skew is visible. | Build the one-layer TP/EP correctness gate and preserve per-rank timing. |
| 2026-05-23 | Hard cut to TP/EP-only implementation work. | Sprint 225 showed the frozen PP path is correct but bottlenecked by layer-scheduled pipeline bubbles. User directed zero further PP variant work. | Sprint 226 starts the TP-only planner and topology contract. |
| 2026-05-23 | Deferred MTP until after TP/EP serving. | MTP can be useful only after the serving runtime has the right topology and multi-slot decode behavior. | Revisit in Sprint 236 or equivalent after TP/EP serving exists. |

## Open Questions

1. Does TP8 remain the primary target after the first collective and expert
   gates, or should TP4/EP8 be used as a temporary correctness stepping stone?
2. Should the first TP/EP pack preserve current TurboMind expert layout exactly
   or repack experts for EP ownership immediately?
3. What correctness tolerance is acceptable for TP/EP low-bit reductions versus
   the frozen PP baseline?
4. Should the first serving target be strictly `32` slots / `256K`, or should
   the TP runtime also gate `32` slots / `128K` as a faster iteration target?
