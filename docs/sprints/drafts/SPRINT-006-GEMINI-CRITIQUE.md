# SPRINT-006 Gemini Critique

## Strong points

- **Rigorous Format Policy:** Both drafts correctly identify the V100 "compute desert" (no native BF16/FP8/FP4). The Claude draft’s approach of encoding this as a fail-closed `ds4_v100_classify` function is the correct engineering path. It turns architectural constraints into runtime errors rather than silent performance regressions or precision loss.
- **Sidecar Architecture:** The decision to implement `ds4_v100_context` as a sidecar to the existing legacy globals is the right risk-mitigation strategy. It allows for the bring-up of an 8-GPU runtime without destabilizing the current single-device GGUF/Metal paths used for development and CPU-side testing.
- **HC Relay Robustness:** The inclusion of a pinned-host fallback for the HC relay primitive is excellent. While peer-to-peer (P2P) is the production requirement for the V100 pod, having a validated host-bounce path ensures the developer loop isn't blocked by single-GPU dev environments or transient P2P failures in the cluster.
- **Layer Skeleton Walker:** The concept of a "no-math" walker that validates descriptor presence and ownership across all 43 layers (in stage order) provides the necessary "connective tissue" verification before the high-complexity kernel sprints (007+) begin.

## Blocking concerns

- **Memory Reserve Guardrails:** While the Claude draft mentions a 256 MiB reserve, the V100-SXM2-32GB has very little margin for error when 1M context slots are active. The "STOP" condition for residency must be more aggressive: if the combined (Weight Arena + KV Arena + Scratch + Relay + CUDA overhead) footprint leaves less than 1 GiB of "true" free HBM per GPU, the sprint should fail. We cannot afford fragmentation-induced OOMs during long-context decode.
- **Descriptor Completeness vs. SHIP:** The Codex draft suggests "one representative family per layer" might be enough for SHIP. This is a risk. We should require *full* descriptor coverage for at least one stage (e.g., Stage 0) to prove that the SWA-only, global embedding, and normal transformer layer types all bind correctly. Partial binding across all stages is less valuable than full binding for a vertical slice of the pipeline.
- **Initialization Mode Ambiguity:** `DS4_V100_INIT_USE_EXISTING_ARENAS` must be the primary validation path on the cluster. If the context relies on re-uploading 150+ GiB of shards on every smoke test, the iteration cycle will be too slow. The smoke tool must verify the *integrity* of existing arenas (size and checksum/magic) before claiming success.

## Recommended merge changes

- **Merge Context API from Claude:** The opaque struct approach and the explicit `ds4_v100_context_options` in the Claude draft should be the foundation of the merged plan.
- **Strengthen Policy Classifier:** Add a Phase 1 requirement to implement `ds4_v100_classify_or_die()`. In a runtime appliance, there is no "maybe" for compute formats. If a tensor-family/dtype pair isn't in the V100-approved list, the process must terminate during initialization.
- **Formalize the "No-Math" Contract:** The layer skeleton walker must explicitly *not* link against or call any `ds4_gpu_attention_*` or `ds4_gpu_moe_*` symbols. This ensures we don't accidentally pull in premature compute code that hasn't been V100-tuned.
- **Expand the Topology Report:** The context report must explicitly emit the "PCI Bus ID -> Visible Device ID -> Stage ID" mapping. Debugging multi-GPU relay is impossible without a clear map of the NVLink/PCI topology.
- **Source-Layout Guard:** Explicitly include the "Source-model generation remains guarded" check in the final Definition of Done for all phases.

## Deferred items

- **KV Allocation Logic:** The context should own the *reserve* for KV, but the actual allocation and slot-management logic should remain deferred to Sprint 008 (long-context/scheduler).
- **HMMA Kernels:** All dequant-to-FP16 HMMA logic is strictly deferred. This sprint ends at the point where a kernel *could* be launched with a valid descriptor and a valid scratch pointer.
- **MTP / Speculative Decoding:** No state for MTP should be allocated in this context yet, although `gpu7`'s reserve must account for its future presence.
- **Server Integration:** `ds4_server.c` should not be touched. The V100 context is a CLI/runtime primitive for now.
