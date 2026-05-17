# SPRINT-001 Critique: Claude vs Gemini

This critique is focused on the stated goal of Sprint 001: a kill-gated
feasibility sprint for an 8x V100-SXM2-32GB DS4 appliance, not a full
performance port.

A few repo-grounded observations matter before comparing the drafts:

- The current CUDA backend is in fact single-device today: `ds4_cuda.cu`
  still carries global model/cache/cuBLAS state and initializes only device 0.
- The current public GPU surface in `ds4_gpu.h` is intentionally narrow and
  opaque, so any multi-GPU plan needs to preserve that discipline or justify
  widening it.
- `tests/cuda_long_context_smoke.c` is useful regression coverage, but it does
  not exercise model loading, layer placement, or cross-device handoff. Any
  real sprint draft needs additional verification, not just reuse of that test.

## Executive Verdict

The Claude draft is the stronger foundation by a wide margin. It understands
the sprint as a bounded feasibility spike, answers most of the intent's open
questions, defines explicit kill gates, and ties the work to the current code
shape.

The Gemini draft is directionally correct but too thin to run as the actual
sprint plan. It reads more like an executive summary than an execution-ready
specification.

If one draft is chosen as the base, it should be the Claude draft, but trimmed
and corrected in a few places noted below.

## Claude Draft

### Strengths

- It is tightly aligned with the intent: fit and correctness first, no NCCL,
  no speculative decoding, no broad kernel import, and a hard stop/go outcome.
- It correctly identifies the main architectural problem in the current repo:
  single-device global CUDA/cuBLAS/model-cache state in `ds4_cuda.cu`.
- The phased plan is practical. P0 through P5 form a sensible escalation:
  build, preserve single-device behavior, add plan abstraction, validate
  copy/path mechanics, attempt real model fit, then run a minimal decode.
- The draft makes the model-format decision explicit. That is important because
  the intent names format choice as a first-class fork decision.
- Its Definition of Done is substantially better than the Gemini draft because
  it requires evidence artifacts, kill gates, and a real q2-imatrix fit test.
- The risk matrix is the only one of the two drafts that treats single-device
  regression as a top-tier risk, which is correct.

### Weaknesses

- It is over-specified in places that do not need to be fixed at sprint-planning
  time. The exact helper names, struct shapes, output formats, tag name, and
  report filenames make the draft heavier than a feasibility sprint needs.
- It introduces several non-core deliverables that are useful but not essential
  to the feasibility answer: `--cuda-info`, a dedicated inventory document,
  a followups document, a branch tag, and a project-memory update.
- It leaks too much placement policy into the public GPU surface. Adding
  `ds4_gpu_tensor_alloc_on()`, `ds4_gpu_layer_device()`, and similar helpers in
  `ds4_gpu.h` may be justified, but the draft does not explain why these must
  be public API rather than CUDA-internal plumbing.
- Its memory budgeting in P4 is stated with too much confidence for a sprint
  whose purpose is to measure feasibility. The estimates are useful planning
  numbers, but they should not read like established facts.
- The security section is much more elaborate than the sprint warrants. The
  real missing rigor is in runtime correctness and memory-accounting edge cases,
  not network or telemetry analysis.

### Gaps In Risk Analysis

- The draft does not adequately call out cross-device synchronization as a
  first-order correctness risk. A valid multi-GPU design needs more than
  `cudaMemcpyPeerAsync`; it needs explicit stream/event ordering when layer N
  on device A hands HC state to layer N+1 on device B.
- It underplays the risk that "fit" could be reported via managed-memory or
  host-mapped behavior that hides true device residency limits. The sprint
  needs a crisp policy on whether managed KV or host-backed fallback still
  counts as success.
- It does not explicitly treat tensor aliasing helpers such as
  `ds4_gpu_tensor_view()` as a migration risk. Once tensors have device
  ownership, base/view consistency becomes a real source of silent bugs.
- It does not elevate q8 preload/cache duplication to a major risk. A plan
  that shards layer weights can still lose feasibility if caches are mirrored
  or staged redundantly per device.
- It mentions homelab workload contention only late as a surfaced question. For
  an 8x V100 appliance sprint, competing VRAM consumers on `gpu-01` should be a
  named operational risk, not a footnote.

### Missing Edge Cases

- Subset visibility cases need stronger treatment: `CUDA_VISIBLE_DEVICES=0`,
  `0,1`, and `0..7` should all be part of the expected regression matrix.
- Forced host-staged fallback should be tested deliberately, not only observed
  if peer access happens to fail on the hardware.
- The draft does not explicitly say how `ds4_gpu_tensor_alloc_managed()` will
  interact with per-device ownership and fit accounting.
- It does not specify how `ds4_gpu_tensor_contents()`, `begin_commands()`,
  `flush_commands()`, and `end_commands()` behave once multiple devices and
  streams exist.
- It covers malformed manual device plans, but not malformed or surprising
  intersections between `CUDA_VISIBLE_DEVICES` and `DS4_DEVICE_VISIBLE`.

### Definition Of Done Completeness

- The DoD is mostly strong and materially better than Gemini's.
- The single-device byte-identical `--dump-logprobs` gate is a good bar and
  should stay.
- The DoD should add one explicit requirement that the multi-GPU path works
  correctly under at least 1-GPU, 2-GPU, and 8-GPU visible-device scenarios.
- The DoD should add one explicit requirement that the recorded fit result say
  whether it was achieved with pure device residency, managed KV, host staging,
  or any other fallback.
- The administrative close-out items should be demoted. A branch tag and
  followup doc are useful, but they should not sit at the same importance level
  as "q2-imatrix actually fits and decodes."

## Gemini Draft

### Strengths

- It identifies the right core themes: SM70 buildability, per-device state,
  layer sharding, HC transfer, smoke testing, and a real model-load attempt.
- It stays focused on the appliance idea instead of drifting into broad
  framework work.
- Its shorter format makes the main point easy to read quickly.

### Weaknesses

- It is too shallow to serve as the sprint spec. There is no meaningful phase
  structure, no execution sequence, and no explicit stop/go flow.
- It does not resolve the format decision even though the intent makes that one
  of the central questions.
- It introduces an inappropriate first-sprint success criterion:
  "performance or simplicity advantage over `llama.cpp`." The intent clearly
  says Sprint 001 is a feasibility slice, not a performance bake-off.
- It proposes a new `docs/sprints/VISION.md`, which is unnecessary scope for a
  bounded feasibility sprint.
- It says "extend `ds4_gpu.h` with device/layer awareness" without enough
  precision about what must stay narrow versus what can become public surface.

### Gaps In Risk Analysis

- It almost completely misses the biggest engineering risk in this repo:
  preserving the current single-device CUDA behavior while splitting global
  backend state.
- It does not discuss model-cache corruption, pointer ownership, or cache
  duplication risks at all.
- It does not discuss cross-device synchronization semantics, only copy
  mechanics.
- It does not mention the risk that peer access may be partial rather than
  uniformly available.
- It does not mention operational risks around staging an 81 GiB model and
  securing uninterrupted access to all eight GPUs.

### Missing Edge Cases

- No coverage of 43-layer distribution edge cases across different visible
  device counts.
- No explicit handling of embeddings, output head, and other non-per-layer
  tensors.
- No explicit negative testing for malformed device plans or visibility masks.
- No explicit force-test for host-staged fallback.
- No treatment of managed-memory behavior and whether it counts as a valid
  feasibility outcome.
- No requirement to keep `make cuda-regression` and the existing single-device
  smoke path healthy throughout the sprint.

### Definition Of Done Completeness

- The DoD captures some of the obvious artifacts, but it is incomplete as a
  kill-gated feasibility contract.
- It lacks a hard single-device equivalence gate.
- It lacks a concrete stop condition if q2 only fits with unacceptable context
  reduction or host-backed fallbacks.
- It lacks a required report of actual per-device placement and memory usage.
- It lacks verification of manual override behavior and device summary output.
- The `<30GB usage per GPU at 64k context` line is not grounded in the intent
  and is likely the wrong feasibility threshold for Sprint 001.

## Recommendation

Use the Claude draft as the base, but tighten it in four ways:

- Trim non-essential sprint outputs so the plan stays centered on feasibility
  evidence, not operator UX or process artifacts.
- Add explicit coverage for stream/event ordering, tensor-view/device-metadata
  consistency, managed-KV policy, and forced host-staged fallback testing.
- Add a small visible-device regression matrix: 1 GPU, 2 GPUs, and 8 GPUs.
- Make the fit verdict explicit about what kind of residency made success
  possible. "It fits" is not enough if the mechanism is managed memory or host
  staging that would invalidate the appliance claim.

The Gemini draft is still useful as a short summary, but it is not sufficient
as the sprint document without being expanded almost to the shape Claude already
provides.
