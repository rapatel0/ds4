# Overview

Sprint 001 is a bounded feasibility spike for turning this private `ds4` fork
into a DeepSeek V4 Flash appliance for the homelab 8x V100-SXM2-32GB stack.
The goal is not to ship a finished replacement for the current
`llama.cpp`/TurboMind path. The goal is to answer one question with hard
evidence: can DS4's narrow, fixed-graph design support a credible 8-GPU V100
runtime using the existing DS4 q2 path, or should this fork stop before more
time is spent?

The sprint stays intentionally narrow:

- Primary target model: DS4-compatible `q2-imatrix` or `q2` GGUF.
- Primary hardware target: 8 visible `sm_70` V100 GPUs with 32 GB each.
- Primary architectural bet: contiguous layer sharding with per-device CUDA
  ownership and 64 KiB HC handoff between device boundaries.
- Explicit non-goals: MXFP4/FP8 loader support, speculative decoding, server
  concurrency, NCCL collectives, and broad kernel import from `deepseek`.

This sprint is kill-gated. Passing means DS4 can enumerate and own all visible
GPUs, produce a deterministic 43-layer placement plan, perform correct
cross-device HC relay, and show a plausible q2 fit/load path across 8 devices.
Failing any of those gates is an acceptable sprint outcome if the failure is
documented precisely enough to justify stopping the fork.

# Use Cases

- As the operator, I can build the CUDA backend on the V100 host with
  `make cuda CUDA_ARCH=sm_70` and see explicit `sm_70` readiness or explicit
  build blockers.
- As the operator, I can run a CUDA regression that reports visible devices,
  peer-access availability, per-device memory, and a verified 64 KiB HC copy
  path across at least two GPUs.
- As the runtime, I can compute and print a deterministic 43-layer placement
  plan for 8 GPUs, with an optional bring-up override for contiguous splits.
- As the runtime, I can allocate graph state, model-cache state, scratch, and
  cuBLAS handles per device without leaking global state across devices.
- As the evaluator, I can inspect per-device planned bytes before a full load
  attempt and decide whether q2 is plausible on 8x 32 GB V100 without guessing.
- As the decision maker, I get a stop/go verdict based on measured fit and
  correctness evidence, not on architectural optimism.

# Architecture

The first-sprint architecture should preserve DS4's narrow public surface and
move multi-GPU ownership behind the existing tensor-resident GPU API.

- Device runtime:
  `ds4_gpu_init()` should enumerate all visible CUDA devices and build one
  internal runtime context per device. Each context should own its own
  `cudaSetDevice` scope, `cublasHandle_t`, model-range cache, q8 preload cache,
  temporary buffers, upload stream, and accounting counters. The current global
  CUDA state in `ds4_cuda.cu` becomes per-device state.
- Tensor ownership:
  `ds4_gpu_tensor` should carry its owning device internally. Same-device
  operations remain unchanged at call sites. Cross-device copies should flow
  through `ds4_gpu_tensor_copy()`, which chooses `cudaMemcpyPeerAsync` when peer
  access is available and falls back to a pinned host bounce buffer when it is
  not. Bounds and ownership checks must fail closed.
- Active-device control:
  `ds4.c` should switch devices at graph boundaries, not by adding device
  arguments to every math primitive. A narrow control surface such as
  `ds4_gpu_set_active_device()` is acceptable; broad per-kernel device plumbing
  is not.
- Layer placement:
  Add a `ds4_layer_device_plan` owned by `ds4.c`. The default planner should
  produce a contiguous 43-layer split across visible GPUs, reserve headroom on
  the final device for output/logits state, and keep embeddings on device 0 and
  the output head on the last device. A bring-up-only env override such as
  `DS4_CUDA_LAYER_SPLIT=6,6,6,5,5,5,5,5` is acceptable because it preserves one
  contiguous execution model instead of introducing permanent runtime variants.
- Inter-device handoff:
  The only inter-device payload in Sprint 001 is HC state. That boundary is
  small enough to keep the first design simple:
  `DS4_N_HC * DS4_N_EMBD * sizeof(float) = 64 KiB`. No NCCL is required in this
  sprint. Layer-local KV and scratch remain resident on the owning device.
- Model-format gate:
  Sprint 001 should use DS4's existing q2/q4 loader and type system. The
  current `DSv4-Flash-256e-fixed.gguf` MXFP4/FP8 path remains out of scope. If
  the DS4 q2 path cannot fit or cannot produce a coherent load path across 8
  GPUs, the sprint stops. It does not expand into format-port work mid-sprint.

# Implementation

## Phase 0: Build And Fit Baseline

- Build on the CUDA host with `make cuda CUDA_ARCH=sm_70`.
- Capture whether `tests/cuda_long_context_smoke` still builds and runs before
  any multi-GPU changes.
- Record the target DS4-compatible model choice for the sprint:
  `q2-imatrix` first, `q2` second, `q4` optional.
- Add a lightweight byte-estimation path in `ds4.c` that reports:
  per-layer weight bytes, embedding/output bytes, estimated graph scratch,
  estimated KV bytes at the chosen context, and per-device planned totals.

Verification:

- `make cuda CUDA_ARCH=sm_70`
- `make cuda-regression`

Kill gate:

- Stop if `sm_70` support fails for reasons deeper than a localized build fix.
- Stop if there is no DS4-compatible q2 model available for a real fit attempt.

## Phase 1: Split CUDA Runtime State Per Device

- Refactor `ds4_cuda.cu` so the current globals become fields on a device
  context array indexed by visible device.
- Extend `ds4_gpu_init()` to:
  enumerate devices;
  create per-device cuBLAS handles;
  print a per-device capability and memory table;
  precompute peer-access reachability.
- Preserve the existing single-device behavior when only one device is visible
  or when the layer plan resolves entirely to device 0.
- Extend `ds4_gpu_print_memory_report()` so it reports per-device totals for:
  cached model bytes, q8 cache bytes, scratch bytes, and free/total memory.

Verification:

- `CUDA_VISIBLE_DEVICES=0 ./tests/cuda_long_context_smoke`
- `CUDA_VISIBLE_DEVICES=0,1 ./tests/cuda_long_context_smoke`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tests/cuda_long_context_smoke`

Kill gate:

- Stop if the refactor breaks the single-device CUDA path or leaves hidden
  shared state that makes device ownership non-deterministic.

## Phase 2: Add Layer Planning And Device-Aware Allocation

- Add a `ds4_layer_device_plan` in `ds4.c` with:
  visible device count;
  per-layer owner device for all 43 layers;
  embedding device;
  output device;
  estimated bytes per device.
- Use the plan during graph allocation so layer-local state tensors and model
  cache ownership resolve to the correct device before decode begins.
- Reserve the last device for output/logit work and let the planner shift one or
  more trailing layers left if needed to keep headroom on that device.
- Keep the override surface narrow and contiguous only. The sprint should not
  introduce arbitrary non-contiguous layer maps.

Verification:

- Print the resolved plan at startup.
- Verify that the estimated q2 resident bytes plus scratch/KV stay credibly
  below 32 GB per device before attempting a full load.

Kill gate:

- Stop if q2 still does not plausibly fit after contiguous sharding and last
  device reservation.
- Stop if the planner requires a non-contiguous placement scheme to fit.

## Phase 3: Implement Cross-Device HC Relay

- Add device ownership metadata to `ds4_gpu_tensor`.
- Teach `ds4_gpu_tensor_alloc()` and `ds4_gpu_tensor_alloc_managed()` to allocate
  on the current active device.
- Teach `ds4_gpu_tensor_copy()` to choose:
  same-device copy;
  peer copy with `cudaMemcpyPeerAsync`;
  pinned host-bounced copy when peer access is unavailable.
- Extend `tests/cuda_long_context_smoke.c` with a new HC-sized copy check that:
  allocates buffers on two devices;
  writes a deterministic pattern;
  copies device A -> device B -> host;
  verifies exact bytes.

Verification:

- `CUDA_VISIBLE_DEVICES=0,1 ./tests/cuda_long_context_smoke`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tests/cuda_long_context_smoke`

Kill gate:

- Stop if HC relay is not exact.
- Stop if cross-device copy requires NCCL or a larger architectural change just
  to move 64 KiB between adjacent layer shards.

## Phase 4: Integrate A First Layer-Sharded Decode Skeleton

- Update the GPU graph allocation in `ds4.c` so:
  embeddings seed HC on device 0;
  each layer runs on its planned device;
  HC copies occur only at device boundaries;
  the output head runs on the final device.
- Keep the first pass minimal:
  no speculative decode;
  no server batching work;
  no new quant formats;
  no imported TurboMind dense path.
- Run a short prompt smoke with a DS4-compatible q2 model and record:
  startup plan;
  per-device memory before/after graph allocation;
  per-device memory after first prompt prefill;
  exact load success or exact failure point.
- If q2 load succeeds, run a short decode probe and compare the behavior against
  the nearest DS4-compatible reference path available. If full token-level
  comparison is not yet practical, require exact device-local invariants plus a
  non-garbage decode trace before continuing the fork.

Verification:

- `make cuda-regression`
- short-prompt q2 load/decode on the 8x V100 host with all devices visible

Kill gate:

- Stop if DS4 cannot load or keep q2 resident across the 8x 32 GB plan.
- Stop if the first sharded decode path is immediately incoherent and the cause
  is broader than device ownership or HC relay.
- Stop if the only way forward is to port MXFP4/FP8 support before DS4 proves a
  viable q2 appliance path.

## Phase 5: Close With A Stop/Go Verdict

- If all prior gates pass, close the sprint with a recommendation to continue
  into correctness-hardening and performance work.
- If any gate fails, close the sprint with an explicit stop/defer decision and
  record the blocker that made DS4 lose to the current llama.cpp path.

Verification:

- Sprint closeout must include the commands run, device visibility, chosen
  model, per-device memory numbers, and the exact gate that produced the
  decision.

Kill gate:

- The sprint is considered successful if it produces a defensible stop/go
  verdict. It is not required to force a “go” outcome.

# Files Summary

- `ds4_gpu.h`
  Add the narrow multi-GPU control surface, keep tensor internals opaque, and
  preserve the tensor-resident execution model.
- `ds4_cuda.cu`
  Replace single-device globals with per-device runtime state, add device
  enumeration, peer/fallback copy logic, and per-device memory reporting.
- `ds4.c`
  Add the layer placement planner, per-device byte estimation, graph-boundary
  device switching, and HC relay at layer-device boundaries.
- `tests/cuda_long_context_smoke.c`
  Extend the existing CUDA regression with device enumeration and HC cross-copy
  validation so multi-GPU bring-up remains inside the normal CUDA test path.
- `Makefile`
  Only touch this if the existing `cuda-regression` target cannot cleanly carry
  the new smoke coverage.
- `README.md`
  Update only if the sprint clears the go gate, documenting `sm_70` build usage,
  the q2-first appliance posture, and the current limits of the V100 path.
- `CONTRIBUTING.md`
  Update only if the sprint clears the go gate, documenting the required
  multi-GPU regression commands and what evidence must accompany future CUDA
  changes.

# Definition of Done

Sprint 001 is done only when it produces a hard decision backed by reproducible
evidence.

- `make cuda CUDA_ARCH=sm_70` succeeds, or a concrete `sm_70` blocker is
  isolated and judged sprint-ending.
- DS4 prints all visible CUDA devices and a per-device memory/capability report.
- DS4 prints a deterministic contiguous 43-layer placement plan for the visible
  devices and estimated bytes per device.
- `tests/cuda_long_context_smoke` validates exact HC-sized cross-device copy on
  the visible topology, using peer access when possible and host bounce
  otherwise.
- CUDA state ownership is per-device for cuBLAS, model caching, q8 caching, and
  scratch accounting.
- A real q2 fit/load attempt is run on the 8x V100 host and records success or
  exact failure with per-device memory numbers.
- The sprint closes in one of two states:
  `GO` if q2 fit/load and the first sharded decode skeleton are credible.
  `STOP` if a kill gate is hit and the evidence shows DS4 is not yet worth
  pursuing versus the existing llama.cpp path.

# Risks & Mitigations

- Risk: the current global CUDA state is more entangled than expected and the
  refactor destabilizes even single-device runs.
  Mitigation: preserve single-device mode as the first regression target and do
  not merge later phases until Phase 1 passes `cuda-regression`.
- Risk: the final device runs out of memory because it carries too many layers
  plus output/logit buffers.
  Mitigation: make the planner byte-aware and allow only contiguous left-shifts
  away from the final device.
- Risk: peer access is unavailable or asymmetric on the V100 topology.
  Mitigation: support pinned host-bounced HC relay from the start; 64 KiB is
  small enough that this is acceptable for the sprint.
- Risk: q2 weight residency still blows the memory budget once caches and KV are
  counted.
  Mitigation: estimate bytes before full load, treat q2 as the required target,
  and stop instead of broadening scope into q4 or format-port work.
- Risk: the temptation to import TurboMind or `tc-grid` kernels expands the
  sprint into performance work before fit and correctness are proven.
  Mitigation: keep all kernel-import work explicitly deferred until after the
  q2 multi-GPU baseline exists.
- Risk: the current operational model format mismatch pressures the sprint into
  MXFP4/FP8 support work.
  Mitigation: make the q2 DS4 path the only in-scope model path for Sprint 001
  and use a stop verdict if that path is not viable.

# Security Considerations

- Preserve DS4's strict GGUF validation. Sprint 001 must not loosen tensor-name,
  dimension, or type checks just to make more files load.
- Parse any layer-split override strictly and fail closed on malformed input,
  wrong layer counts, or device ids outside the visible set.
- Keep cross-device copy helpers bounds-checked and ownership-checked so a bad
  planner state cannot silently write to the wrong device buffer.
- Do not expand the server/API surface in this sprint. Multi-GPU bring-up should
  stay in the local runtime and test path only.
- Treat memory reports, trace logs, and model paths as local debugging output.
  They should remain opt-in and should not be emitted to remote clients.

# Dependencies

- An 8x V100-SXM2-32GB Linux host with a working CUDA toolchain and `nvcc`
  capable of `CUDA_ARCH=sm_70`.
- Access to a DS4-compatible `q2-imatrix` or `q2` GGUF and enough local disk
  plus host memory for mmap-backed loading.
- The existing DS4 CUDA build and regression path:
  `make cuda CUDA_ARCH=sm_70` and `make cuda-regression`.
- The current DS4 source surfaces:
  `ds4.c`, `ds4_gpu.h`, `ds4_cuda.cu`, and `tests/cuda_long_context_smoke.c`.
- Reference-only context from `/Users/ravi/repos/deepseek`, especially the
  TurboMind MXFP4 fix, the routed-expert grouped-compare tests, and
  `tools/tc-grid` sm70 notes. These are inputs to decision-making, not build
  dependencies for Sprint 001.
- Peer access between some or all GPUs is helpful but not mandatory. Host bounce
  is an acceptable fallback in this sprint.

# Open Questions

- Is the required success bar for Sprint 001 a full q2 short decode on 8 GPUs,
  or is a successful fit/load plus exact HC relay sufficient to continue?
- Should the sprint require `q2-imatrix` specifically, or may plain `q2` count
  as the baseline if it is easier to stage on the cluster first?
- How much headroom should the planner reserve on the final device for output
  and logits before it starts shifting layers left?
- Is a contiguous split override enough for the homelab stack, or is there a
  real reason to support non-contiguous placement later?
- If q2 fits but decode is still incoherent, what is the next proof step:
  two-layer sharded correctness harness, layer-by-layer tensor dumps, or an
  early stop verdict?
- If q2 proves viable, should Sprint 002 prioritize correctness hardening of
  the q2 appliance path or model-format work toward the current MXFP4/FP8
  operational model?
