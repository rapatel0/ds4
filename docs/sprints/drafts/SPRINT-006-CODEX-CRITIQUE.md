# SPRINT-006 Codex Critique
## Strong points

- The intent and layout documents are pointed in the right direction: Sprint 006 sits between pack residency/probe work and decode correctness, keeps cross-GPU traffic to HC boundaries, keeps activations FP16, and keeps the source-model generation guard closed.
- The Claude draft is much closer to a production-worthy V100 plan than the Gemini draft. It correctly centers Sprint 006 on a sidecar context, typed descriptors, an HC relay primitive, and a no-math layer skeleton instead of trying to "almost run" decode.
- The Claude draft also gets the V100 numeric contract mostly right: no native BF16, FP8, or FP4 tensor-core execution; BF16 remains a source/probe format; FP8 and MXFP4 remain packed source/runtime inputs to future kernels rather than executable math formats.
- Using a standalone context smoke tool and synthetic CUDA tests is the right verification shape for this sprint. It allows topology, memory, descriptor, and relay validation without pulling `ds4.c` into a decode rewrite.

## Blocking concerns

- The Gemini draft overreaches badly on scope. Replacing raw GGUF offsets "at call sites," modifying `ds4.c` as part of the main path, and requiring a full 1328-tensor skeleton validation turns Sprint 006 into a broad runtime refactor instead of a bounded context/policy/relay/skeleton sprint.
- The merged plan must be stricter about V100 math policy. "FP16 for compute" is not enough. It should explicitly say: production dense GEMMs on V100 are FP16 HMMA with FP32 accumulation; broad FP32 GEMMs are not an acceptable default; BF16 is source/probe/explicit-conversion only; FP8/FP4/MXFP4 are not executable math formats unless a validated unpack/dequant or low-bit kernel family is registered.
- The intent and Gemini draft still leave decode creep on the table through phrases like "layer scheduler," "owns per-GPU weight arenas," "KV cache," and "every tensor required by the model." Sprint 006 should not claim KV population, RoPE append, decode sequencing, output-head execution, real MoE/router execution, MTP, or server work.
- Fail-closed behavior is underspecified outside the Claude draft. The merged plan needs an explicit STOP if visible GPUs and PCI identity cannot be mapped cleanly onto the 8-stage layout.
- The merged plan also needs an explicit STOP if scratch, relay, CUDA overhead, and borrowed-arena footprint push any GPU below reserve.
- Relay validation must fail closed on invalid GPU ids, self-copy, `active_slots` overflow, or unsupported transfer mode instead of silently degrading behavior.
- Descriptor binding must fail closed on missing pack entries, source-dtype mismatches, wrong owning GPU, duplicate/conflicting bindings, or contradictions with `DS4-V100-LAYOUT.md`.
- The source-layout generation guard needs its own explicit fail-closed check so the new context cannot accidentally open a decode path.
- Even the Claude draft should be tightened on descriptor scope. Sprint 006 should validate the global embedding, HC control families, and one representative full layer span. It should not imply near-complete binding coverage for all layers or all execution call sites.
- "Source-layout embedding dtype fix or replacement path" does not belong in the merged Sprint 006 plan unless it is reduced to report-only classification work. As written, it reads like decode bring-up leakage.

## Recommended merge changes

- Use the Claude draft as the base document. Keep the sidecar `ds4_v100_context*` boundary, standalone smoke tooling, and minimal optional inspect-only plumbing. Do not make `ds4.c`, `ds4_gpu.h`, or the legacy CUDA globals the main implementation surface for this sprint.
- Replace any merged language about a "layer scheduler" with "layer skeleton walker." This sprint should walk, classify, and validate. It should not schedule, enqueue, or simulate decode execution.
- Tighten the numeric policy in the final plan: FP16 HMMA with FP32 accumulation for production dense math on V100; FP32 for control, reductions, norms, router scores, and debug; BF16 only for source-faithful storage, probes, and explicit conversion boundaries; FP8 / MXFP4 / FP4 only as packed source/runtime inputs to validated future kernels; and explicit rejection of broad FP32 GEMMs or any "native BF16/FP8/FP4 compute" language.
- Narrow the Sprint 006 descriptor requirement to metadata that proves the contract rather than metadata that pretends decode is ready. The ship bar should be: global embedding, representative F32 control tensors, representative FP8 and MXFP4 families as descriptors only, HC control tensors, and one representative full layer row set.
- Add hard kill gates to the merged plan: STOP if the 8-stage topology cannot be mapped cleanly onto the visible V100 set; STOP if any GPU drops below declared reserve after context allocations; STOP if any in-scope descriptor family resolves to `UNSUPPORTED` or contradicts `DS4-V100-LAYOUT.md`; and STOP if source-model generation progresses further than it did before Sprint 006.
- Keep the real-pack smoke flexible: `probe-only` and `use-existing-arenas` are valid Sprint 006 bring-up modes. Full resident decode-ready wiring is not a Sprint 006 ship requirement.
- Remove or defer any merged language that implies output-head math, KV population, MoE execution, MTP/spec decode, tensor-parallel exceptions, or server deployment.

## Deferred items

- Single-slot decode correctness and any normal source-model generation path.
- KV allocation/population, RoPE append, compressed-KV/indexer flows, and long-context slot admission.
- Real MoE/router execution, grouped MXFP4 or INT8 expert kernels, FP8 dense kernels, and shared-expert math.
- Output-head execution, vocab parallelism, and other tensor-parallel exceptions.
- `ds4_server.c` integration, appliance serving flow, and deployment/runtime operations.
