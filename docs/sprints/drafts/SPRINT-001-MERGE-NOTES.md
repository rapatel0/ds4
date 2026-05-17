# SPRINT-001 Merge Notes

## Source Drafts

- `SPRINT-001-CLAUDE-DRAFT.md`
- `SPRINT-001-CODEX-DRAFT.md`
- `SPRINT-001-GEMINI-DRAFT.md`
- `SPRINT-001-CLAUDE-CRITIQUE.md`
- `SPRINT-001-CODEX-CRITIQUE.md`
- `SPRINT-001-GEMINI-CRITIQUE.md`

## Interview Refinements

The final sprint diverges from all three drafts on the model target. The drafts
treated DS4 q2-imatrix as the first target because that is what upstream DS4
already supports. The user clarified that the appliance should maximize model
intelligence while using quantized DSv4 models. The working source target is
`/models/DSv4-Flash-256e-fixed.gguf` or an equivalent high-intelligence
quantized DSv4 file, but the appliance runtime should pack into a small set of
V100-friendly layouts: FP8 first, and INT8 where the existing integer kernels
validate accuracy and speed.

The user also clarified:

- Pure device residency is required for GO. Fast SSD or host staging may be used
  only as an explicitly labeled fallback/diagnostic path, not the default
  appliance behavior.
- A full short decode is the GO bar. Fit/load plus hidden-context relay is an
  EXTEND result because it does not prove the runtime produces correct tokens.
- Stop when there is a material area of uncertainty with no improvement, rather
  than spending time on a blocked path by default.
- Defer `VISION.md` until feasibility is understood.

Follow-up refinement after reviewing the current TurboMind/tc-grid performance
state:

- The observed llama.cpp/TurboMind full-model numbers are single-slot,
  layer-scheduled, and do not include DS4-style MTP decode.
- Sprint 001 should treat those numbers as the current feasibility floor, not
  as the appliance ceiling.
- MTP, multi-slot batching, tensor-scheduled hot ops, LM-head splitting, and
  expert scheduling are expected future uplift paths once coherent single-slot
  appliance-pack decode exists.

Follow-up refinement after choosing a quantized runtime-pack strategy:

- Separate source support from runtime layout. MXFP4/F8 GGUF support is needed
  to read and validate the model, but raw MXFP4 does not have to be the final
  kernel layout.
- FP8 is the quality-first runtime pack target. INT8 is a candidate where the
  measured V100 integer kernels can carry the tensor family and calibration
  preserves decode quality.
- tc-grid and prior integer-kernel work move from background evidence to a
  first-class candidate path, but broad experimental kernel import remains
  deferred behind a narrow tensor-family pack boundary.
- INT8 must be opt-in per tensor family until scale policy, checksum coverage,
  and reference/decode comparisons pass.

Follow-up refinement after correcting the DeepSeek4 KV cache estimate:

- Use the `llama-memory-deepseek4.cpp` formula
  `kv_size = n_swa + (ratio ? n_ctx_seq / ratio : 0)`, with layers 0-1
  SWA-only, 21 ratio-4 layers with indexer, and 20 ratio-128 layers without
  indexer.
- At 1M context, F16 KV/cache state is planned as roughly 8-10 GiB aggregate
  per slot, not the earlier conservative 26 GiB estimate. F8 is roughly
  4-5 GiB; F32 debug is roughly 16-18 GiB.
- On an 8-GPU layer split, a single 1M F16 slot is memory-feasible. The real
  admission-control constraint is multi-slot long context because KV state
  multiplies by slot.
- Sprint 001 now requires a static `ds4-v100-plan` style report that prints
  layer placement, per-GPU weight/KV/scratch/reserve, kernel family choices,
  peer matrix, and admitted slots for 128K, 256K, 512K, and 1M tiers.

## Accepted Draft Strengths

- From Claude: concrete per-device CUDA state analysis, explicit model-format
  decision point, bit-equivalence/no-regression posture, peer/host HC-copy
  fallback, and SHIP/EXTEND/STOP close-out structure.
- From Codex: early byte-estimation gate, clean kill-gate sequencing, narrow
  public API posture, and explicit pure-residency reporting.
- From Gemini: concise framing around appliance feasibility and the reminder
  that managed memory is a possible diagnostic fallback, not a default success
  path.

## Accepted Critiques

- Add explicit stream/event ordering risk for cross-device HC handoff.
- Add tensor-view/device-metadata consistency as a first-class risk.
- Add 1-GPU, 2-GPU, and 8-GPU visible-device regression coverage.
- Require the fit report to say whether weights, KV, and scratch are pure device
  resident, managed, host-staged, or mmap/host-backed.
- Add forced host-staged fallback testing for the HC relay.
- Add GGUF SHA-256 and tensor inventory to prevent phantom model results.
- Do not create `VISION.md` in this sprint.

## Rejected Draft Points

- Rejected q2-imatrix as the primary Sprint 001 target. It is deferred as a
  fallback/reference path because the appliance should target the
  highest-intelligence quantized DSv4 source with FP8/INT8 runtime packing.
- Rejected a performance-vs-llama.cpp gate for Sprint 001. Performance is
  measured, but correctness plus pure VRAM short decode is the first gate.
- Rejected NCCL as Sprint 001 scope. The layer-boundary payload is small enough
  for peer copy or host bounce while proving feasibility.
- Rejected server concurrency, speculative decoding/MTP, tensor scheduling, and
  broad kernel import as first-sprint deliverables. These remain future uplift
  paths rather than reasons to judge Sprint 001 performance pessimistically.

## Final Synthesis

Sprint 001 is a quantized-pack, pure-VRAM feasibility sprint. It first
inventories and budgets the quantized DSv4 source model using the corrected
DeepSeek4 compressed KV layout, defines the FP8-first/INT8-candidate runtime
pack policy, then adds the minimum DS4 loader/packer support and 8-GPU CUDA
ownership needed for a short decode. It can close as:

- `SHIP`: the FP8-first or validated INT8 appliance pack fits in pure VRAM
  across 8x V100 and a short decode is coherent.
- `EXTEND`: model format and residency are substantially proven, but decode is
  blocked by a bounded fix.
- `STOP`: a material uncertainty remains with no improvement, or pure VRAM
  residency/short decode is not credible.
