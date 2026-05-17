---
sprint: 001
title: Quantized FP8/INT8 DS4 Appliance Feasibility on 8x V100
status: planned
date: 2026-05-17
target_repo: rapatel0/ds4
merge_notes: drafts/SPRINT-001-MERGE-NOTES.md
---

# SPRINT-001: Quantized FP8/INT8 DS4 Appliance Feasibility on 8x V100

## Overview

This sprint tests whether the private `ds4` fork can become a narrow appliance
for the highest-intelligence DeepSeek V4 Flash local model on the homelab
8x V100-SXM2-32GB stack. The target is a quantized DSv4-Flash source model
packed into a small set of V100-friendly runtime layouts: FP8 first for quality
and INT8 where the existing integer-kernel work can validate accuracy and speed.
The upstream DS4 q2/q4 family remains a fallback/reference path, not the primary
appliance target.

The goal is not a complete server or a speed win yet. The goal is a hard
feasibility answer: can DS4 load the quantized source model, build a compact
FP8/INT8 appliance pack, keep packed weights resident in VRAM across eight
32 GB cards, run the fixed DS4 graph across those cards, and produce a coherent
short decode? If not, the sprint closes with the specific blocker instead of
drifting into an unbounded port.

The current llama.cpp/TurboMind measurements are a floor, not the expected
ceiling for a DS4 appliance. They are single-slot, layer-scheduled runs with no
MTP decode and only routed experts accelerated by TurboMind. Sprint 001 should
record its first decode speed against that context, but the sprint outcome is
still feasibility and correctness. MTP, multi-slot batching, tensor-scheduled
hot ops, LM-head splitting, and tighter expert scheduling are follow-on uplift
paths, not prerequisites for the first coherent DS4 decode.

Outcome contract:

- `SHIP`: the quantized source model is converted or packed into an FP8-first
  or validated INT8 appliance layout, pure device resident across 8x V100, and
  a short greedy decode is coherent.
- `EXTEND`: format support, residency, and HC relay are proven, but decode is
  blocked by a bounded issue with a clear next fix.
- `STOP`: a material uncertainty remains with no improvement, pure VRAM
  residency is not credible, or short decode cannot be made coherent without
  expanding into a different runtime strategy.

Non-goals:

- No q2/q4 appliance path as the primary target.
- No raw MXFP4 runtime requirement if offline conversion into FP8/INT8 proves
  simpler and preserves quality.
- No SSD/offload default path.
- No NCCL, tensor-parallel row split, expert-parallel routing, server
  concurrency, speculative decoding, MTP, LM-head split, or tensor-scheduled
  performance work as Sprint 001 deliverables.
- No upstream PR.
- No `VISION.md` until feasibility is understood.

## Use Cases

1. **Quantized Model Appliance Proof**: Run the highest-intelligence quantized
   DeepSeek V4 Flash model from a DS4-specific FP8/INT8 appliance runtime on
   the 8x V100 stack.
2. **Pure VRAM Fit Proof**: Place model weights, layer-local KV, and required
   graph state on GPUs without relying on SSD or host-backed weight streaming as
   the default path.
3. **Runtime Feasibility Verdict**: Produce a reproducible SHIP/EXTEND/STOP
   report based on exact model file, tensor inventory, per-device memory, and a
   short decode result.
4. **Performance Floor Context**: Label the first DS4 decode result as a
   single-slot feasibility baseline and preserve the expected follow-on uplift
   paths for later sprints.
5. **Planner Contract**: Produce a dry-run 8x V100 placement/admission report
   that uses the DeepSeek4 compressed KV layout, not a generic full-KV memory
   estimate.

## Architecture

### Source Model And Runtime Pack

Primary source target:

- `/models/DSv4-Flash-256e-fixed.gguf`, or the equivalent high-intelligence
  quantized DeepSeek V4 Flash source file.

Required source tensor formats:

- `GGML_TYPE_MXFP4 = 39` for FP4/MXFP4 blocks.
- `GGML_TYPE_F8_E4M3_B128 = 42` for FP8 E4M3 blocks with E8M0 scale per 128
  values.

Sprint 001 must not loosen DS4's fixed DeepSeek shape validation. It should add
an explicit source-format policy table derived from the target model inventory:
which tensor families are F16/F32/Q8_0, which are FP8, which are MXFP4, and
which are legal to convert offline into INT8. Unexpected types or dimensions
remain fatal.

Runtime pack targets:

- **FP8-first**: preferred initial weight layout for dense/shared/attention
  tensors because it is closest to the source quantized model and minimizes
  quality risk.
- **INT8 candidate**: preferred where existing V100 integer kernels already
  have good measured behavior, especially routed experts or other large GEMMs,
  but only after calibration and decode-quality checks pass.
- **F16 reference/fallback**: allowed for small metadata tensors or for
  correctness isolation, but not as a default full-weight layout.

MXFP4 support remains necessary for reading and validating the source model,
but Sprint 001 should not require raw MXFP4 to be the final runtime kernel
format if an offline FP8/INT8 pack gives simpler memory layout and better V100
kernel coverage.

### Multi-GPU Ownership

`ds4_cuda.cu` is single-device today: it initializes device 0, keeps global
cuBLAS/model-cache/q8-cache/scratch state, and has no layer ownership. Sprint
001 introduces per-visible-device CUDA state while keeping DS4's public API
narrow.

Each visible device owns:

- cuBLAS handle and math-mode state;
- model/tensor cache ranges for layers assigned to that device;
- optional packed FP8/INT8 kernel metadata;
- scratch and upload/prefetch streams;
- memory accounting.

The runtime must work with `CUDA_VISIBLE_DEVICES=0`, `0,1`, and all eight
visible GPUs. Logical visible-device IDs are used internally; physical IDs are
reported for diagnostics only.

### Layer Plan

Default placement is contiguous layer sharding across visible GPUs. For 43
layers on 8 GPUs, the planner must produce a deterministic, memory-aware map
and report it at startup. It should optimize for pure VRAM fit, lowest
worst-device memory, enough reserve for slots/context, few boundary crossings,
and output-head imbalance. Embeddings live on the first layer's device; the
output head lives on the final layer's device unless byte budgeting proves a
different contiguous placement is needed.

The only inter-device activation in Sprint 001 is the DS4 hidden-context state:

`DS4_N_HC * DS4_N_EMBD * sizeof(float) = 4 * 4096 * 4 = 64 KiB`

HC transfer uses `cudaMemcpyPeerAsync` when a directed peer pair is available
and a pinned host bounce buffer otherwise. Host bounce for HC is acceptable
because it is tiny; host-backed weights are not a GO condition.

### Appliance Planner And Pack Contract

Plan the runtime as a static DS4 appliance scheduler, not as ad hoc kernel
calls. Layer ownership is the base layout; tensor-scheduled exceptions are
future performance work only where they clearly pay off.

The first pack contract should be an offline per-GPU shard layout:

```text
ds4-v100-pack/
  manifest
  gpu0.weights
  gpu1.weights
  ...
  gpu7.weights
```

Each packed tensor descriptor should include semantic tensor ID, original dtype
and shape, packed dtype/layout, owning GPU, layer ID, kernel family, byte
offset, checksum, and scale-buffer offset when applicable.

Runtime memory is split into four arenas per GPU:

- **Weight arena**: immutable, packed at conversion time, pure device resident,
  with no duplicate raw GGUF bytes in VRAM.
- **KV arena**: layer-owned and slot-owned as
  `[local_layer][slot][cache_row][dim]`; raw SWA stays tiny while compressed KV
  grows with context.
- **Scratch arena**: reused across local layers and sized by active microbatch,
  not total configured slots.
- **Relay arena**: double-buffered HC transfer as
  `[2][active_slots][4][4096]` F32, using peer async when possible and pinned
  host bounce only for the 64 KiB HC payload when necessary.

The near-term diagnostic contract is:

```bash
ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off
```

It should print the layer map, per-GPU weight bytes, per-GPU KV bytes,
scratch/reserve/headroom, selected kernel family per tensor group, directed
peer matrix, and admitted max slots for 128K, 256K, 512K, and 1M contexts.
This planner output becomes the contract for the packer and runtime.

### Context, Slots, And KV Admission

Use the DeepSeek4 KV layout from `llama-memory-deepseek4.cpp`, not a generic
full-context KV estimate:

`kv_size = DS4_N_SWA + (ratio ? n_ctx_seq / ratio : 0)`

Compression schedule:

- layers 0-1: ratio 0, SWA-only, 128 slots regardless of context;
- even layers 2,4,...,42: ratio 4 with indexer, 21 layers;
- odd layers 3,5,...,41: ratio 128 without indexer, 20 layers.

At `n_ctx_seq = 1,048,576` and F16 cache elements, the planning estimate is:

- `attn_kv`: about 5.41 GiB aggregate;
- ratio-4 `indexer_kv`: about 1.31 GiB aggregate;
- F32 compression-state buffers: rough envelope 0.5-1.5 GiB aggregate;
- decode scratch/activations: about 1-2 GiB at M=1 decode, with larger
  transient during 1M prefill.

Therefore the single-slot aggregate planning envelope is about 8-10 GiB for
F16 KV, about 4-5 GiB for F8 KV, and about 16-18 GiB for F32 debugging. With
8-GPU layer ownership, F16 1M KV is roughly 0.7-1.2 GiB per device depending on
the layer map. A single 1M slot is memory-feasible if weights are balanced; the
admission-control risk is multi-slot long context because each slot owns its
own positional KV state.

Planning caveats:

- `--parallel 8` multiplies the KV side by 8, so 8 slots at 1M context is not a
  practical default. Multi-slot operation should use shorter context tiers.
- Ratio-4 layers dominate long-context KV memory and bandwidth because
  `n_ctx_seq / 4` dwarfs the ratio-128 layers.
- At 1M decode, long-context attention is bandwidth-bound: each token touches
  compressed KV in addition to the weight stream, so MTP launch amortization
  does not erase the HBM limit.

Recommended initial slot modes:

| Mode | Slots | Context | MTP | Purpose |
|------|------:|--------:|-----|---------|
| `feasibility` | 1 | 4K -> 32K | off | prove coherent decode |
| `long` | 1-2 | 512K -> 1M | optional later | max context |
| `balanced` | 4 | 256K | after validation | first aggregate tok/s target |
| `throughput` | 8 | 128K -> 256K | after validation | aggregate tok/s |
| `latency` | 1 | 32K -> 256K | optional later | minimal queueing |

Configured slots and active microbatch must be separate. The appliance may
configure 8 slots but run active batches of 4 if that gives the best
latency/throughput balance.

### Kernel Strategy

Correctness comes first, but the appliance should prefer quantized runtime
layouts that match the available V100 kernels. Use the prior DeepSeek work as
source material:

- MXFP4 nibble mapping must match the fixed TurboMind handoff:
  low nibble maps to `k = j`, high nibble maps to `k = j + 16` within each
  32-value block.
- Routed expert MXFP4 can be decoded or converted offline into FP8/INT8 pack
  rows; the runtime should prefer the measured grouped sm70/integer paths where
  they validate.
- Single dense FP8/MXFP4 TurboMind remains opt-in only; prior work showed that
  TurboMind single dense tensors can be numerically unsafe for shared experts.
  Dense FP8 support must either match the llama.cpp native CUDA path closely or
  stay in a correctness-first fallback path until measured.
- INT8 execution is allowed only with explicit scale policy, checksum coverage,
  and decode-quality comparison against the FP8/source reference. It should be
  treated as a pack format decision, not an implicit lossy rewrite inside the
  runtime.

Kernel selection should be a registry keyed by tensor family and shape:

| Tensor family | First-choice kernel | Notes |
|---------------|---------------------|-------|
| Routed experts | INT8 or FP8 grouped sm70 | Prefer measured integer kernels if quality passes |
| Dense/shared tensors | FP8 first, INT8 candidate | Avoid unsafe TurboMind single dense by default |
| Attention projections | FP8/F16 reference, INT8 candidate | Correctness first |
| LM head | FP8/INT8 layer-owned initially | Later vocab split only if memory/latency requires it |
| MTP support | Separate optional module | Enable after base decode |

tc-grid and the existing integer kernels are now first-class implementation
evidence for the INT8 candidate path. Do not broadly import experimental
kernels without a narrow pack/kind boundary, and do not make INT8 the default
for a tensor family until its calibration and decode-quality checks pass.

### Baseline And Uplift Context

The latest measured llama.cpp/TurboMind context from
`/Users/ravi/repos/deepseek` is:

- Full 8x V100 DSv4-Flash-256e default CUDA, layer split: about 11.35 tok/s
  decode and 9.66 tok/s prefill on a short probe.
- Full 8x V100 routed-expert TurboMind after the MXFP4 nibble-lane fix: about
  13.09 tok/s on the 96-token Fibonacci probe.
- Broad `exps|shexp` safe mode: about 12.32 tok/s, because shared experts fall
  back to default CUDA rather than the unsafe single dense TurboMind path.

Those numbers do not include the DS4-specific advantages that remain plausible:

- **MTP/speculative decode**: DS4 already has MTP concepts in the runtime
  surface, while the llama.cpp path has to retrofit DeepSeek4 partial-tail
  memory semantics.
- **Higher effective M**: multi-slot batching or speculation can move expert
  GEMMs away from the `M=1` regime where both TurboMind and tc-grid are least
  efficient.
- **Tensor-scheduled hot ops**: layer scheduling fits the model, but serializes
  whole layers. A DS4 appliance can later split selected heavy operations while
  keeping the rest of the graph layer-owned.
- **LM-head and expert scheduling**: output projection and routed experts are
  first-class DS4 graph nodes, so future sprints can split or fuse them without
  changing a generic runtime scheduler.

Sprint 001 should not be judged as a failure for lacking these uplifts. It
should fail only if quantized source loading, FP8/INT8 packing, pure VRAM
residency, cross-device execution, or coherent short decode are not credible.

## Implementation

### Phase 0: Inventory, Build, Planner, And Byte Budget (~10%)

**Files:**

- `docs/sprints/SPRINT-001-REPORT.md` — create during execution for command
  logs, tensor inventory, model SHA, memory budget, and verdict.
- `ds4.c` or small local diagnostic helper — add or reuse GGUF inventory
  reporting if needed.
- `ds4_cli.c` or `tools/ds4-v100-plan.c` — add dry-run placement and slot
  admission reporting if cleaner than embedding it in the main CLI.
- `Makefile` — touch only if `make cuda CUDA_ARCH=sm_70` needs a target fix.

**Tasks:**

- [ ] Compute SHA-256 and size for the target quantized DSv4 source model.
- [ ] Inventory tensor names, dimensions, and GGML type IDs; classify tensor
  families into F16/F32/Q8_0/FP8/MXFP4 and candidate FP8/INT8 runtime pack
  layouts.
- [ ] Define the initial pack policy: FP8-first by default, INT8 only for tensor
  families with an explicit scale/calibration policy and reference comparison.
- [ ] Build with `make cuda CUDA_ARCH=sm_70`.
- [ ] Run existing CUDA smoke with one visible GPU.
- [ ] Estimate per-device bytes for a contiguous 8-GPU layer plan:
  weights, packed metadata, KV at test context, graph scratch, cuBLAS overhead,
  embeddings, and output head.
- [ ] Use the DeepSeek4 compressed KV formula and report F16/F8/F32 KV
  envelopes for 128K, 256K, 512K, and 1M contexts.
- [ ] Add or sketch `ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off`
  output: layer map, per-GPU bytes, reserve/headroom, kernel families, peer
  matrix, and admitted max slots by context tier.
- [ ] Record whether the plan can fit with pure device-resident weights.

**Kill gate:**

- Stop if the target quantized model file is unavailable or cannot be identified
  exactly.
- Stop if the byte budget shows pure VRAM residency is not credible on 8x 32 GB.
- Stop if sm70 build failure is broad and not improving after a localized fix.

### Phase 1: Quantized Loader, Pack Policy, And Format Validation (~20%)

**Files:**

- `ds4.c` — add source type info and strict quantized-format validation.
- `ds4_gpu.h` — add only narrow declarations needed by the CUDA backend.
- `ds4_cuda.cu` — add block/layout helpers and FP8/INT8 pack upload stubs.
- `tests/` — add focused source/pack layout tests if not covered by CUDA smoke.

**Tasks:**

- [ ] Add `GGML_TYPE_MXFP4 = 39` and `GGML_TYPE_F8_E4M3_B128 = 42` to DS4's
  GGUF type table with correct block sizes and row sizes.
- [ ] Add explicit DS4 source-format tensor policy validation from Phase 0
  inventory.
- [ ] Reject arbitrary quantized formats. Only the target DeepSeek V4 Flash
  layout and the chosen FP8/INT8 pack policy are in scope.
- [ ] Add MXFP4/F8 source decode helpers and FP8/INT8 pack helpers required by
  the CUDA kernels.
- [ ] Define INT8 scale granularity per tensor family, including checksum and
  reference-output coverage.
- [ ] Add tests that catch the MXFP4 nibble-lane bug fixed in the DeepSeek
  TurboMind path.

**Kill gate:**

- Stop if the target model's tensor layout differs from DS4's fixed graph in a
  way that requires a general loader or llama.cpp-style graph framework.

### Phase 2: Per-Device CUDA State And Pure-Residency Placement (~20%)

**Files:**

- `ds4_cuda.cu` — refactor global CUDA state into per-device state.
- `ds4_gpu.h` — expose minimal device/plan diagnostics.
- `ds4.c` — create and consume the layer placement plan.
- `tests/cuda_long_context_smoke.c` or `tests/cuda_multi_device_smoke.c` — add
  1/2/8 visible-device coverage.

**Tasks:**

- [ ] Enumerate visible CUDA devices and print physical ID, name, sm version,
  memory, and directed peer-access matrix.
- [ ] Split cuBLAS handles, model caches, q8/fp8/int8/mxfp4 source-pack caches,
  streams, scratch, and accounting per logical device.
- [ ] Preserve single-device CUDA behavior under `CUDA_VISIBLE_DEVICES=0`.
- [ ] Implement deterministic contiguous `layer_device[43]` planning.
- [ ] Keep KV allocation layer-owned and slot-owned so long context is admitted
  by explicit `ctx * slots` accounting, not hidden global allocation.
- [ ] Report planned and actual per-device resident bytes.
- [ ] Fail closed on malformed layer plans or device IDs outside the visible
  set.

**Verification:**

- `CUDA_VISIBLE_DEVICES=0 ./tests/cuda_long_context_smoke`
- `CUDA_VISIBLE_DEVICES=0,1 ./tests/cuda_long_context_smoke`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tests/cuda_long_context_smoke`

**Kill gate:**

- Stop if single-device CUDA regresses.
- Stop if per-device caches duplicate enough data to break pure VRAM residency.

### Phase 3: Cross-Device HC Relay And Stream Ordering (~10%)

**Files:**

- `ds4_cuda.cu`
- `tests/cuda_long_context_smoke.c` or `tests/cuda_multi_device_smoke.c`

**Tasks:**

- [ ] Add device metadata to `ds4_gpu_tensor` and propagate it to tensor views.
- [ ] Implement exact 64 KiB HC copy for same-device, peer-copy, and forced
  host-bounce paths.
- [ ] Add stream/event ordering so device B cannot consume HC before device A's
  producing work completes.
- [ ] Test directed peer availability and forced host-bounce fallback.
- [ ] Run all-pairs or at least boundary-pair HC relay validation for the
  planned 8-GPU layer map.

**Kill gate:**

- Stop if HC relay is not exact or cannot be ordered without a larger runtime
  redesign.

### Phase 4: FP8/INT8 CUDA Execution Path (~25%)

**Files:**

- `ds4_cuda.cu`
- optional imported/adapted files from `/Users/ravi/repos/deepseek/ggml/vendor/turbomind`
  only if they can be kept isolated behind DS4's CUDA backend.
- optional isolated integer-kernel files adapted from prior `tc-grid` or
  TurboMind experiments, behind the same pack/kind boundary.
- focused tests under `tests/`

**Tasks:**

- [ ] Wire routed expert execution through the selected FP8/INT8 pack path,
  preferring the measured sm70 grouped/integer kernels where quality passes.
- [ ] Wire dense/shared FP8 tensors with a path that matches the reference
  numerical contract closely enough for short-decode coherence.
- [ ] Add INT8 dense/shared candidates only after the scale policy and focused
  reference comparisons pass.
- [ ] Keep TurboMind single dense execution disabled unless a focused test
  proves it safe.
- [ ] Add shape tests for DS4 decode dimensions:
  routed gate/up, routed down, shared gate/up/down, attention/output projections.
- [ ] Validate outputs against llama.cpp/TurboMind or small host references
  where feasible.

**Kill gate:**

- Stop if the only working execution path depends on host-streamed weights,
  unexplained numerical drift, or an unvalidated INT8 conversion.
- Stop if dense FP8/shared-expert execution or INT8 scale selection becomes a
  material uncertainty with no improving test signal.

### Phase 5: Layer-Sharded Graph Integration And Short Decode (~15%)

**Files:**

- `ds4.c`
- `ds4_cli.c` if a diagnostic flag is needed
- `docs/sprints/SPRINT-001-REPORT.md`

**Tasks:**

- [ ] Seed embeddings on the first device.
- [ ] Run each layer on its planned device.
- [ ] Transfer HC only at device boundaries.
- [ ] Keep layer-local KV on the owning device.
- [ ] Run output head on the planned output device.
- [ ] Attempt a short greedy decode with the FP8-first or validated INT8
  appliance pack.
- [ ] Compare against the existing llama.cpp/TurboMind baseline where possible:
  at minimum non-empty UTF-8 coherent text, no repeating token loop, and no
  obvious code/completion collapse on a fixed prompt.
- [ ] Record prefill/decode tokens/sec as a single-slot feasibility baseline,
  not a Sprint 001 gate.
- [ ] Explicitly label missing MTP, multi-slot batching, tensor scheduling,
  LM-head split, and expert scheduling as follow-on uplift paths in the report.
- [ ] Record whether the corrected 1M single-slot KV envelope remains
  memory-feasible on the actual 8x V100 placement.

**GO bar:**

- Quantized DSv4 source model is converted or packed into FP8-first or validated
  INT8 runtime layout and pure device resident across 8x V100.
- A short greedy decode completes coherently.
- The report records model SHA, tensor inventory, device plan, memory use,
  fallback use, command lines, and output sample.

**EXTEND bar:**

- Format support, pure residency, and HC relay are proven, but decode fails due
  to a bounded kernel/ordering issue with a clear next fix.

**STOP bar:**

- A material area of uncertainty remains with no improvement.
- Pure VRAM residency is not credible.
- Short decode cannot be made coherent without changing the strategy.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `ds4.c` | Modify | Quantized source validation, pack policy, byte planning, layer-device graph scheduling |
| `ds4_gpu.h` | Modify | Minimal diagnostics/control surface for multi-device CUDA |
| `ds4_cuda.cu` | Modify | Per-device state, FP8/INT8 pack support, HC relay, sm70 execution |
| `ds4_cli.c` | Modify if needed | Diagnostic flags only, such as model inventory or device report |
| `tools/ds4-v100-plan.c` | Create if cleaner | Dry-run layer placement, KV admission, pack contract reporting |
| `Makefile` | Modify if needed | Preserve/build `make cuda CUDA_ARCH=sm_70` and CUDA smoke targets |
| `tests/cuda_long_context_smoke.c` | Modify | Extend CUDA smoke for 1/2/8 GPUs and HC relay |
| `tests/cuda_multi_device_smoke.c` | Create if cleaner | Focused multi-device placement/copy/format smoke |
| `docs/sprints/SPRINT-001-REPORT.md` | Create during execution | Commands, model SHA, memory, decode sample, verdict |
| `docs/sprints/SPRINT-001-FOLLOWUPS.md` | Create during execution if needed | Execution-discovered follow-ups |

## Definition of Done

- [ ] Target quantized DSv4 source model is identified by path, size, and
  SHA-256.
- [ ] Tensor inventory confirms the exact source-format and runtime-pack policy.
- [ ] Planner reports corrected DeepSeek4 KV budgets for 128K, 256K, 512K, and
  1M, including slot multiplication.
- [ ] `make cuda CUDA_ARCH=sm_70` builds or a precise sm70 blocker is recorded.
- [ ] Existing single-device CUDA smoke remains healthy.
- [ ] DS4 reports visible CUDA devices, directed peer matrix, and memory.
- [ ] Layer placement for 43 layers is deterministic and printed.
- [ ] Per-device CUDA state owns cuBLAS, caches, streams, scratch, and memory
  accounting.
- [ ] Pure device-resident weight placement is demonstrated or disproven.
- [ ] HC relay is exact for same-device, peer-copy, and forced host-bounce
  paths.
- [ ] FP8/INT8 execution tests cover MXFP4 source nibble mapping, pack
  conversion, dense/shared/attention shapes, and INT8 scale policy where used.
- [ ] A short greedy decode either succeeds coherently or fails with a bounded
  blocker.
- [ ] `SPRINT-001-REPORT.md` records SHIP/EXTEND/STOP with commands and evidence.
- [ ] `SPRINT-001-REPORT.md` compares any TPS number to the current
  llama.cpp/TurboMind single-slot layer-scheduled floor without treating that
  floor as the appliance ceiling.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Native mixed-format layout differs from DS4's fixed graph | Medium | High | Phase 0 tensor inventory before runtime work; fail closed on unexpected layout |
| FP8/INT8 support expands into a general quantization framework | Medium | High | Add only source-format and appliance-pack policy tables, not generic graph/type handling |
| Single-device CUDA path regresses during per-device refactor | Medium | High | Keep 1-GPU smoke and existing CUDA regression as Phase 2 gates |
| Pure VRAM fit is hidden by managed memory or host-backed mapping | Medium | High | Report residency class; GO requires device-resident weights |
| Cross-device HC copy races due to missing stream/event ordering | Medium | High | Add explicit event handoff and forced fallback tests |
| MXFP4 nibble-lane bug reappears | Medium | High | Add test that encodes low=`k=j`, high=`k=j+16` mapping |
| Dense FP8/shared-expert path has unexplained numerical drift | High | High | Keep TurboMind single dense disabled; compare to native CUDA/reference outputs |
| INT8 conversion preserves layout but harms quality | Medium | High | Make INT8 opt-in per tensor family until calibration and decode-quality gates pass |
| q8/fp8/mxfp4 cache duplication breaks VRAM budget | Medium | High | Per-device cache accounting before full model load |
| Slot admission uses stale generic KV math | Medium | High | Use DeepSeek4 ratio schedule and report `ctx * slots` KV budgets explicitly |
| 1M context fits in memory but is bandwidth-bound | High | Medium | Treat 1M as long-context mode, not aggregate tok/s mode; measure decode separately |
| Homelab GPU contention blocks 8-GPU experiments | Medium | Medium | Treat as operational blocker in report; do not weaken GO bar |
| Single-slot layer-scheduled baseline understates DS4 appliance upside | Medium | Medium | Separate feasibility TPS from follow-on MTP, batching, tensor scheduling, LM-head, and expert-scheduling work |

## Security Considerations

- Keep strict model layout validation. Do not make arbitrary GGUFs load.
- Parse any diagnostic layer/device override fail-closed.
- Bounds-check all tensor copies and ensure tensor views inherit device
  ownership.
- Do not expose memory reports, local paths, or model SHA through server APIs in
  this sprint.

## Dependencies

- Private repo `rapatel0/ds4`.
- 8x V100-SXM2-32GB node with CUDA toolchain capable of `sm_70`.
- Quantized DeepSeek V4 Flash source model, expected at
  `/models/DSv4-Flash-256e-fixed.gguf` or equivalent.
- Prior reference material in `/Users/ravi/repos/deepseek`:
  `SPRINT-025-TURBOMIND-HANDOFF.md`, `SPRINT-025-V100-SPIKE-DIRECTION.md`,
  `SPRINT-026.md`, `SPRINT-027-LM-HEAD-VOCAB-TP-SPIKE.md`,
  `ggml/vendor/turbomind`, and `ggml/include/ggml.h`.

## Open Questions

- Which exact quantized source model path should be canonical for the appliance
  after Sprint 001?
- What short prompt set should define the coherence gate beyond a Fibonacci
  code probe?
- If dense FP8 or INT8 scale selection is the first bounded blocker, should
  Sprint 002 port llama.cpp's native CUDA F8 path, harden the integer kernels,
  or implement a DS4-specific fused dense path?
- If output head memory pressures the final GPU, should Sprint 002 use
  contiguous layer rebalancing or vocab tensor parallelism?
- After the first coherent DS4 decode, should the next performance sprint
  prioritize MTP, multi-slot batching, LM-head split, tensor-scheduled dense
  ops, or expert-path fusion?
- Should the default appliance expose F16 KV only at first, or include F8 KV as
  an explicit long-context memory/performance mode once correctness is proven?
