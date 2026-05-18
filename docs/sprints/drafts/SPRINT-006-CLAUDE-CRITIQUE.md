# SPRINT-006 Claude Critique

Reviewer perspective: senior CUDA/runtime engineer for the 8x V100-SXM2-32GB
DSv4-Flash appliance. Anchored on `docs/architecture/DS4-V100-LAYOUT.md`, the
INTENT document, and the V100 reality that there is no native BF16, FP8, or FP4
tensor-core compute path. Production GEMMs must be FP16 HMMA with FP32
accumulation, or validated low-bit/integer custom kernels. BF16 stays
source/probe/explicit-conversion only.

## Strong points

**Both drafts**
- Correctly identify the deliverable as structural (context + descriptors +
  relay + no-math walk), not decode.
- Keep the source-model generation guard active. Neither claims runnable decode.
- Reuse the baseline 8-stage contiguous layer map from `DS4-V100-LAYOUT.md`
  rather than redesigning topology.
- Adopt `[active_slots][4][4096]` as the HC boundary payload shape, consistent
  with the architecture doc.
- Exclude decode kernels, MTP, real MoE, KV population, output-head math, and
  server deployment from scope.

**CODEX draft specifically**
- Encodes execution policy as a data-typed enum
  (`DS4_V100_EXEC_F32_CONTROL`, `DS4_V100_EXEC_F16_HMMA`,
  `DS4_V100_EXEC_LOWBIT_KERNEL`, `DS4_V100_EXEC_DIAGNOSTIC_ONLY`) rather than
  leaving policy as comments. This is the right shape for fail-closed binding
  later.
- Policy table includes an explicit "Forbidden Claim" column that bans treating
  BF16/FP8/FP4 as native V100 tensor-core execution. This is the load-bearing
  guardrail.
- Uses a sidecar module name (`ds4_v100_context`) that keeps the new context
  separate from the legacy global CUDA path (`g_cublas`, `g_cuda_tmp`).
- Explicitly forbids silent host-backed relay success. Same-GPU loopback for
  synthetic only; real multi-GPU validation must be device-to-device.
- Mentions double-buffered relay buffers, consistent with the
  `[2][active_slots][4][4096]` arena shape in `DS4-V100-LAYOUT.md`.
- Calls out SWA-only, ratio-4, and ratio-128 differences in descriptor
  coverage — the three layer classes that have different tensor families.
- Open question 4 surfaces the gpu7 output-head/MTP reserve question early.
- Phase 5 produces a separate guard log artifact (`SPRINT-006-GUARD.log`)
  proving the guard still fires after a successful context bring-up.

**GEMINI draft specifically**
- Clear `SHIP / EXTEND / STOP` outcome contract.
- Concise Non-Goals list — easy to reference when scope creep is proposed.
- Clean policy table covering BF16, FP8/MXFP4, FP16, FP32 roles.

## Blocking concerns

### B1. GEMINI's host-pinned relay fallback violates the appliance contract
GEMINI Architecture > HC Relay Boundary says: *"Prefer `cudaMemcpyAsync` with
Peer-to-Peer (P2P) enabled; fallback to host-pinned relay if P2P is
unavailable."* Risk table also lists this as mitigation. This directly
contradicts:
- INTENT line 92: *"Keep pure device residency. Do not introduce
  managed-memory, host-backed, or SSD-backed successful paths."*
- LAYOUT.md "Ground Rules": cross-GPU payloads stay small and stay on device.

A host-pinned fallback that lets the smoke pass on a degraded topology is
exactly the silent-success failure mode the appliance must reject. CODEX gets
this right: real multi-GPU validation must succeed device-to-device or fail
closed. **Remove the host fallback from the merged plan.** If P2P is
unavailable on a stage boundary, the report should print the peer matrix and
the sprint should exit non-zero, not silently re-route.

### B2. GEMINI overreaches into the engine path
- Phase 4 puts `ds4_gpu_layer_skeleton_walk` integration into `ds4.c` and
  describes "engine wiring."
- Definition of Done says: *"Layer skeleton walks the full 1328-tensor model
  and validates ownership."*

Wiring the skeleton through `ds4.c` puts the no-math walk one bug away from
the decode/prefill orchestration and risks the source-layout generation guard.
Walking the full 1328-tensor inventory is also a model-level enumeration, not
a skeleton-level family-presence check; it pulls the sprint toward exhaustive
descriptor coverage that belongs in a later decode-correctness sprint.

CODEX's approach — a sidecar tool (`tools/ds4-v100-context-smoke.c`)
consuming one narrow diagnostic-only entry point — is the right shape. The
walk should validate **descriptor families per layer class**, not enumerate
every tensor.

### B3. Neither draft requires topology fail-closed checks at context open
The appliance contract assumes exactly eight V100-SXM2-32GB devices. Neither
draft mandates that `ds4_v100_context_open` fail closed on:
- `cudaGetDeviceCount() != 8`
- any device with `cudaDeviceProp.major != 7` (a non-V100 sneaks in)
- VRAM per device below the SXM2-32GB threshold
- inability to enable `cudaDeviceEnablePeerAccess` between stage neighbors
  required by the layer-sharded baseline (0↔1, 1↔2, …, 6↔7)
- mismatched UUIDs vs. the pack-index residency map from Sprint 004

Without these, the context will happily "succeed" on a 4-GPU dev box and the
report will look plausible. The merged plan must list these as explicit
preflight refusals.

### B4. Neither draft fail-closes descriptor / source-dtype / execution-class
mismatch
Encoding execution class as an enum (CODEX) is necessary but not sufficient.
The merged plan must require that:
- a descriptor's `source_dtype` (BF16, F8_E4M3_B128, MXFP4, F32, I32) cannot
  bind to `DS4_V100_EXEC_F16_HMMA` without an explicit conversion-stub field
  saying *which* future kernel performs the unpack;
- pack-row byte length must reconcile against the expected `dim × bytes/value`
  for the declared format (e.g. F8_E4M3_B128 ≈ 1.008 B/value including the
  per-128 scale, MXFP4 ≈ 0.531 B/value), and mismatch is fatal;
- no two stages may claim ownership of the same layer or global tensor id;
- arena offset + byte length must fit inside the owning stage's reported arena
  span.

These are the binding-time fail-closed checks the no-math walk depends on.
Otherwise the walk validates "presence" of garbage.

### B5. Memory budget fail-closed is absent
LAYOUT.md "Proposed Shard Memory Estimate" leaves several GiB of headroom per
GPU after weights + 1M-slot F16 KV + global extras. Neither draft requires
the context to refuse to open when any stage exceeds its budget after
accounting for:
- weight arena (existing residency from Sprint 004)
- scratch budget (this sprint)
- relay buffers (double-buffered FP16, plus optional FP32 debug)
- cuBLAS / CUDA driver overhead
- planned KV reservation (planner field, not allocation)
- gpu7 output-head plus MTP reserve carve-out

The report should print per-stage reserve/headroom and fail when reserve drops
below an explicit threshold (suggest 2-4 GiB per stage).

### B6. GEMINI policy table omits the "Forbidden Claim" rail and KV cache row
The whole point of writing the policy down is to keep a future sprint from
accidentally claiming BF16/FP8/FP4 as a native V100 compute format. GEMINI's
table documents the role but not the prohibition. KV cache row is also
missing entirely — relevant because the KV planner fields land in the
context even though KV allocation does not.

CODEX's policy table is the merge candidate; port it wholesale.

### B7. The Sprint 005 guard test must remain positive
Both drafts assert the source-layout generation guard "remains active." Only
CODEX commits to a separate guard log artifact. The merged plan must require
a positive test that runs `ds4_engine_open()` on the production source-layout
path *after* a successful context smoke and observes the guard firing with
the same failure code Sprint 005 archived. A passive "we did not remove the
guard" assertion is not enough — context bring-up is the most likely vector
for the guard to be regressed.

### B8. Naming risk — `ds4_gpu_context` vs `ds4_v100_context`
GEMINI uses `ds4_gpu_context` and modifies `ds4_gpu.h`/`ds4_cuda.cu` in place.
This widens the existing residency-only arena surface and conflicts with later
tensor-parallel exception scopes (vocab-parallel output head, 2-way TP routed
FFN) that are not strictly per-GPU. CODEX's `ds4_v100_context` in a new
`ds4_v100_context.h/.c` keeps the residency arena API untouched and signals
that this context is V100-specific. **Use the V100-specific name.**

## Recommended merge changes

The final Sprint 006 plan should be the CODEX draft with the following
adjustments:

1. **Module name and file boundary.** Keep `ds4_v100_context.h/.c` as the
   only new module. Do NOT modify `ds4_gpu.h` or the residency arena API.
   `ds4.c`/`ds4.h` changes should be limited to one narrow diagnostic entry
   point that returns the new context — and that entry point should remain
   tool-private (header in `ds4_v100_context.h`, not re-exported from
   `ds4.h`) until Sprint 007 begins real decode wiring (CODEX Open Question 1
   answer: keep tool-private).

2. **Adopt CODEX's execution-class enum and "Forbidden Claim" policy table
   verbatim.** Add a KV-cache row covering planner ownership without
   allocation. Drop GEMINI's host-pinned fallback wording entirely.

3. **Topology fail-closed (new Phase 1 task):**
   - `cudaGetDeviceCount() == 8` required.
   - `cudaDeviceProp.major == 7` for every visible device.
   - VRAM ≥ SXM2-32GB threshold per device.
   - `cudaDeviceCanAccessPeer` true for the seven stage-adjacent pairs (0-1,
     1-2, …, 6-7); call `cudaDeviceEnablePeerAccess` and fail closed on any
     missing edge.
   - Pack-index residency UUIDs match the visible device UUIDs.

4. **Descriptor binding fail-closed (Phase 2):**
   - reject pack-row byte length that does not match the declared source
     format's bytes/value × element count;
   - reject duplicate semantic-tensor-id ownership across stages;
   - reject descriptor offset + bytes > owning arena span;
   - require explicit conversion-stub field when a non-FP16 source dtype is
     bound to `DS4_V100_EXEC_F16_HMMA` (the field can be a string like
     `"fp8_e4m3_b128_unpack_to_fp16_hmma_v1"`; the kernel doesn't exist yet
     but the policy slot must).

5. **Relay primitive (Phase 3):**
   - Double-buffered FP16 relay: `[2][active_slots][4][4096]`, with the
     second slot reserved for future scheduler overlap.
   - FP32 debug mode allocated separately, not as the default.
   - Real cross-stage validation must use peer copy device-to-device; emit
     the full peer matrix; fail closed on any host-backed path.
   - Validate at least one real stage boundary on the 8-GPU pod (CODEX Open
     Question 3 answer: one boundary plus full peer matrix is sufficient for
     this sprint).

6. **Memory budget fail-closed (Phase 1 + Phase 5):**
   - per-stage reserve floor of ≥ 2 GiB after weight arena + scratch budget +
     relay buffers + cuBLAS/CUDA overhead + planned KV reservation;
   - carve out the gpu7 output-head reserve (~1 GiB BF16) and a small MTP
     reserve placeholder, even though neither is enabled (CODEX Open Question
     4 answer: reserve, don't allocate).

7. **Guard positive test (Phase 5):**
   - run a context smoke followed by a normal `ds4_engine_open()` source-
     layout decode attempt and archive the rejection in
     `SPRINT-006-GUARD.log`. The guard must fire with the same Sprint 005
     failure surface, not a new one.

8. **Layer skeleton scope (Phase 4):**
   - validate **descriptor families per layer class** (SWA-only, ratio-4,
     ratio-128), not every tensor. One representative family per layer class
     is enough for `SHIP` (CODEX Open Question 2 answer).
   - the walk emits family presence, ownership, and execution-class
     classification; it must not launch decode kernels or no-op kernels into
     the streams (close GEMINI Open Question 3 the conservative way).

9. **Real cluster smoke (Phase 5):**
   - use bounded "context plus selected spans" mode by default; full
     residency comes from Sprint 004 if needed (CODEX Open Question 5 answer
     and a direct response to GEMINI Open Question 1).
   - archive: topology + memory + peer + policy + relay + guard logs.

10. **Outcome contract.** Adopt GEMINI's SHIP/EXTEND/STOP structure and
    apply it to CODEX's scope. The STOP condition should be "topology, peer,
    or budget fail-closed checks cannot be satisfied on the production pod,"
    not GEMINI's "fundamentally incompatible with `ds4.c`."

11. **Risk table.** Use CODEX's six-row table; add a seventh row: "Sprint
    006 smoke succeeds on a degraded topology (fewer than 8 GPUs, missing
    peer edges, or insufficient VRAM) because preflight checks were
    skipped." Mitigation: the Phase 1 topology fail-closed list above.

## Deferred items

The merged plan must explicitly continue to defer all of these. Any of them
showing up in the implementation is a scope-creep signal.

- **Decode and prefill orchestration.** No attention math, no FFN math, no
  RoPE, no SwiGLU, no router scoring, no output projection, no top-k
  sampling. The source-layout generation guard stays active.
- **KV population.** Planner fields and reserved ownership only; no
  `attn_kv` writes, no indexer_kv writes, no SWA append. F16-vs-F8 KV choice
  also deferred.
- **Real MoE.** No grouped MXFP4 routed-expert kernel, no shared-expert FP8
  kernel, no dequant-to-FP16 tile path, no SwiGLU fusion. Descriptor binding
  for these families is the only sprint-006 work.
- **Output head.** Descriptor binding and reserve carve-out only on gpu7;
  no BF16 → FP16/FP32 projection, no vocab-parallel TP exception.
- **MTP / speculative decoding.** Reserve placeholder on gpu7 only; no
  scheduler, no MTP-specific descriptors, no Q8_0/Q4_K sidecar wiring.
- **Tensor-parallel exceptions.** Vocab-parallel output head, 2-way TP
  routed/shared FFN, and full 2-way TP topology all stay out of scope until
  the layer-sharded baseline has a correctness gate.
- **Server deployment.** `ds4-server`, HTTP/gRPC, batching, slot admission,
  and request scheduling are all out of scope. The context must not be
  reachable through any network surface in this sprint.
- **Persistent dequantized weight copies.** No materialized FP16/F32 copies
  of BF16/FP8/MXFP4 sources at any time, including inside the skeleton walk.
- **Real low-bit kernel implementation.** TurboMind sm70 grouped MXFP4,
  custom FP8 dequant + FP16 HMMA dense kernel, and any integer expert path
  are all later-sprint work. The execution-class enum names the slots; this
  sprint does not fill them.
- **Stream-aware variant of the Sprint 005 BF16 probe.** May be reused as an
  optional descriptor-backed spot check inside the smoke tool, but not as a
  decode entry point and not as a stream-orchestration probe.
- **Long-context slot admission and compression-state buffer planning.**
  Future planner work; not part of the no-math skeleton.
