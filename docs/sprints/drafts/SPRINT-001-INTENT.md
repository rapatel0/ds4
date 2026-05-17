# Sprint 001 Intent: V100 DS4 Appliance Fork

## Seed

The user wants this private fork of `antirez/ds4` to become an appliance for
running DeepSeek V4 Flash on the homelab 8x V100-SXM2-32GB stack. The immediate
motivation is that the llama.cpp fork has achieved useful kernel and TurboMind
progress, but integrating every DeepSeek4 special case into the general
llama.cpp codebase remains difficult. This repo should take the opposite bet:
keep the runtime narrow, DS4-specific, and optimized for the known V100
hardware.

The user specifically asked to review
`/Users/ravi/repos/deepseek/docs/sprints/SPRINT-025-DS4-EVAL.md` and the
kernels created in `/Users/ravi/repos/deepseek`, then sprint-plan the first
private-fork iteration.

## Context

- The local repo is now a private GitHub mirror at `rapatel0/ds4`, with
  `origin` pointing to the private repo and `upstream` preserving
  `antirez/ds4`.
- `ds4.c` is already the right architectural shape for an appliance: fixed
  DeepSeek V4 Flash constants, strict GGUF layout validation, tensor-resident
  GPU graph API, CLI/server/eval tools, disk KV cache, and DS4-specific fused
  CUDA/Metal primitives.
- The CUDA backend is not yet fit for the V100 stack: `ds4_cuda.cu` initializes
  only device 0, uses global CUDA/cuBLAS/model-cache state, and has no
  layer-to-device ownership, peer copies, per-device weight caches, or
  multi-GPU memory report.
- Prior DeepSeek sprint work has a coherent llama.cpp/TurboMind path on 8x V100
  for `/models/DSv4-Flash-256e-fixed.gguf`, including the MXFP4 nibble-lane
  fix, DeepSeek4 slot/KV reset fix, and routed-expert TurboMind path. It also
  shows TurboMind single dense tensors are not numerically safe by default.
- The local `ds4` repo has no `docs/sprints/VISION.md` and no prior local
  sprint docs. Planning starts from scratch here, while borrowing external
  context from `/Users/ravi/repos/deepseek/docs/sprints`.

## Recent Sprint Context

- `SPRINT-025-DS4-EVAL.md` concluded DS4 is strategically interesting but not
  an immediate replacement for llama.cpp because it cannot load the current
  FP4/FP8/MXFP4 GGUF and its CUDA backend is single-device.
- `SPRINT-025-PATCH-REPORT.md` and `SPRINT-025-TURBOMIND-HANDOFF.md` isolated
  and fixed the multi-GPU TurboMind routed-expert gibberish root cause: GGML
  MXFP4 low/high nibbles map to `k=j` and `k=j+16`, not adjacent `2*j` and
  `2*j+1`.
- `SPRINT-025-V100-SPIKE-DIRECTION.md` recommends a bounded DS4 fork spike only
  if the first proof points are concrete: load or convert the current target
  model, create a minimal 8-GPU layer-sharded skeleton, reuse TurboMind or
  equivalent sm70 kernels, and beat or materially simplify the llama.cpp path.
- `SPRINT-026.md` moved the llama.cpp path into native MTP speculative decode
  work and documented DS4 as a useful reference for exact verifier-state and
  compressed KV handling.
- `tools/tc-grid` contains V100-focused sm70 kernel work, including the
  v13_rf_v6 champion and TurboMind design notes. Those kernels are useful for
  a later DS4 performance phase, but the first DS4 sprint must make model fit
  and correctness measurable before importing more kernels.

## Vision Context

No vision document exists in this repo. Planning from scratch.

## Relevant Codebase Areas

- `AGENT.md`: project rules; keep DS4 narrow, readable, C-only except backend
  requirements, correctness before speed, no permanent semantic variants behind
  flags, no C++ in public DS4 code.
- `README.md`: project intent and supported q2/q4 DS4 GGUF family; CUDA is
  supported but originally oriented around DGX Spark/GB10 rather than V100.
- `Makefile`: Linux CUDA builds use `make cuda CUDA_ARCH=sm_70`; `make
  cuda-regression` runs `tests/cuda_long_context_smoke`.
- `ds4.c`: fixed model constants, GGUF layout validation, graph scheduling,
  model tensor caching, decode/prefill layer graph, output head, sessions, and
  cache serialization.
- `ds4_gpu.h`: narrow tensor-resident GPU API. It currently has no device
  argument, device map, or ownership metadata.
- `ds4_cuda.cu`: CUDA backend globals, model-range cache, Q8 preload cache,
  cuBLAS handle, CUDA tensor allocation/copy/read/write, and the fused DS4 CUDA
  kernels.
- `tests/cuda_long_context_smoke.c`: current CUDA regression shape for top-k
  and long-context attention overflow paths. It is small enough to extend for
  multi-GPU API skeleton tests.
- `/Users/ravi/repos/deepseek/ggml/vendor/turbomind/*`: fixed MXFP4
  deinterleave, grouped compare tests, and V100 TurboMind API pieces that may
  become a DS4 routed-expert backend later.
- `/Users/ravi/repos/deepseek/tools/tc-grid/*`: sm70 HMMA kernel experiments,
  especially `mma_sm70.cuh`, `v13_rf_v6`, and TurboMind insight docs.

## Constraints

- Preserve DS4's appliance model: do not make this a generic llama.cpp clone or
  general GGUF runner.
- Keep implementation style compatible with the repo: C public surface,
  CUDA/Objective-C backend files where needed, compact comments near complex
  inference or memory logic, no C++ in DS4 public APIs.
- Correctness gates must precede speed gates. Any V100 path must compare logits
  or token output against the existing DS4 CPU/CUDA behavior or an external
  reference before optimization work is trusted.
- V100 target is 8 GPUs with 32 GB each. A single 81 GB q2 DS4 GGUF cannot be
  resident on one device; first viable architecture is layer sharding.
- The HC activation crossing layer boundaries is small enough to copy between
  devices initially: `DS4_N_HC * DS4_N_EMBD * sizeof(float) = 64 KiB`.
- Model format is a fork decision point. Published DS4 q2/q4 GGUFs are
  easiest to validate against DS4 today; the current llama.cpp operational
  model uses FP4/FP8/MXFP4 tensors and requires new loader/kernel support in
  DS4.
- Do not spend the first sprint on speculative decoding, server concurrency,
  or broad kernel import. Those are follow-on work after a coherent multi-GPU
  DS4 baseline exists.

## Success Criteria

This sprint is successful if it produces a concrete V100-readiness branch with
hard stop/go evidence:

- `make cuda CUDA_ARCH=sm_70` builds on a CUDA host, or build failures are
  isolated to explicit, documented sm70 incompatibilities.
- DS4 reports all visible CUDA devices and prints a per-device memory/capability
  summary.
- A minimal device-plan abstraction exists for 43 layers across 8 GPUs, with
  deterministic contiguous default placement and an override surface suitable
  for the homelab stack.
- CUDA state is split enough that each device can own its own cuBLAS handle,
  model weight cache, scratch accounting, and tensor allocations without
  corrupting the existing single-device path.
- A small multi-GPU smoke test verifies device-local allocation plus a
  cross-device HC-sized tensor copy path.
- The sprint documents the model-format decision with measured evidence:
  published DS4 q2/q4 compatibility path, current llama.cpp FP4/FP8/MXFP4 path,
  or explicit scope/defer choice.
- The Definition of Done includes a kill gate: stop or defer if DS4 cannot
  reasonably load/fit a q2 or converted target model across 8x 32 GB V100.

## Verification Strategy

- Reference implementation: Existing DS4 single-device CUDA behavior for small
  GPU API calls, DS4 CPU/reference math where feasible, and the known coherent
  llama.cpp/TurboMind output only as an external behavioral baseline.
- Spec/documentation: `AGENT.md`, `CONTRIBUTING.md`, `README.md`,
  `SPRINT-025-DS4-EVAL.md`, `SPRINT-025-V100-SPIKE-DIRECTION.md`, and the
  TurboMind handoff docs.
- Edge cases identified:
  - single-device CUDA path must remain usable;
  - 8 visible GPUs may not all support peer access, so fallback host-staged
    copies may be needed;
  - layer plan must handle 43 layers across 8 devices without off-by-one
    placement bugs;
  - global CUDA state must not leak one device's cuBLAS handle, current device,
    model cache, or scratch into another device's call;
  - model cache pressure must be visible before attempting full 81 GB q2 load.
- Testing approach:
  - build: `make cuda CUDA_ARCH=sm_70`;
  - regression: `make cuda-regression`;
  - new tests: multi-device init/plan/copy smoke in `tests/`;
  - cluster proof: run in the homelab V100 pod with `CUDA_VISIBLE_DEVICES=0..7`;
  - model proof: at minimum inspect/attempt load of the selected GGUF and record
    exact failure/success, per-device memory, and next required change.

## Uncertainty Assessment

- Correctness uncertainty: High. DS4's fixed graph helps, but multi-GPU CUDA
  ownership and DeepSeek compressed KV/state are easy to corrupt, and the
  current llama.cpp target model format does not match DS4.
- Scope uncertainty: High. A full appliance is too large for one sprint; this
  sprint must be a kill-gated feasibility slice, not a full performance port.
- Architecture uncertainty: High. The key decision is whether to layer-shard
  the existing q2/q4 DS4 path first, port the current MXFP4 model format first,
  or build a hybrid using TurboMind-style routed-expert kernels.

## Open Questions

1. Should Sprint 001 require loading the published antirez q2-imatrix GGUF
   across 8 GPUs, or is a smaller multi-GPU skeleton plus exact format-failure
   report enough?
2. Is the desired appliance target the published DS4 q2/q4 GGUF family, the
   existing llama.cpp `/models/DSv4-Flash-256e-fixed.gguf`, or both with a
   staged bridge?
3. Should the first multi-GPU implementation shard only layer weights and KV by
   layer, leaving embeddings on device 0 and output head on the last device?
4. Is NCCL required for the homelab V100 stack, or is 64 KiB HC state transfer
   via peer/host `cudaMemcpyPeerAsync` sufficient for Sprint 001?
5. Which existing DeepSeek kernels are worth importing first after fit:
   TurboMind MXFP4 routed experts, tc-grid v13_rf_v6 dense kernels, or DS4's
   existing q2/q4 CUDA MoE path?
6. What is the stop-loss threshold for this fork: one week without coherent q2
   output, failure to fit q2 across 8 GPUs, or failure to outperform/simplify
   the llama.cpp path?
