# SPRINT-001 — DS4 V100 Appliance Feasibility (CLAUDE draft)

**Status:** DRAFT 2026-05-17
**Scope:** kill-gated 8x V100-SXM2-32GB feasibility spike for the private
`rapatel0/ds4` fork
**Predecessor:** none in this repo. External: `deepseek` repo SPRINT-025
(DS4-EVAL, V100-SPIKE-DIRECTION, TURBOMIND-HANDOFF, SPRINT-026)
**Successor:** SPRINT-002 (only if the kill gates in §6 pass) — fused decode on
sharded weights and/or model-format bridge

---

## 1. Overview

This sprint is the first iteration of the private DS4 fork's evolution from a
single-device CUDA backend (DGX Spark / Mac-Metal shape) into an 8x V100
appliance for DeepSeek V4 Flash. It is intentionally narrow: it adds the
*minimum* multi-GPU machinery required to make the existing tensor-resident
graph runnable across 8 devices, and it makes a recorded model-format decision.
It does **not** import new kernels, port FP4/FP8/MXFP4 weights, add NCCL, or
attempt a performance win.

The sprint succeeds when one of the following is provably true:

1. **Go.** A `q2-imatrix` (~81 GiB) DS4 GGUF resides across 8x V100-SXM2-32GB
   with deterministic layer-to-device placement, the single-device CUDA path is
   bit-identical to pre-refactor, and a tiny greedy continuation passes a
   coherence eyeball against the existing CPU reference.
2. **Stop.** Any kill gate (see §6) fails with documented evidence, the fork
   is left in a clean state, and a `STOP` verdict is recorded.

There is no TPS target. The number landed by P5 is the *baseline*, not a gate.
This sprint exists to convert "would DS4 on V100 be useful?" from an
architectural question into a measured one.

**Stop-loss:** one calendar week from start, regardless of phase. (Per intent
open question 6.)

**What this sprint is not:**

- No FP4/FP8/MXFP4 loader, no MXFP4 deinterleave port. The deepseek-fork
  TurboMind path already owns that work; bridging into DS4 is Sprint 002+ if
  this spike survives.
- No NCCL. HC state at layer boundaries is 64 KiB
  (`DS4_N_HC * DS4_N_EMBD * sizeof(float)`); a `cudaMemcpyPeerAsync` or a
  host-staged fallback covers Sprint 001 needs. (Intent Q4.)
- No new fused kernels. The existing `ds4_gpu_*` API is left numerically
  identical; only its ownership story changes.
- No speculative decoding, no server concurrency, no batched prefill changes.
- No upstream PR. All work lands on the private fork.

**Reusable primitive shipped regardless of go/stop:** per-device CUDA state
(cuBLAS handle, model range cache, scratch, math-mode flag) plus a
`layer_device[DS4_N_LAYER]` plan abstraction. Any downstream DS4 multi-GPU work
needs this; a STOP at P4 or later still leaves it.

---

## 2. Use Cases

Each phase has a usable artifact even if the next one slips:

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | `make cuda CUDA_ARCH=sm_70` either succeeds, or its failure is captured as a narrow list of sm70 incompatibilities with reproduction commands. Device inventory dump committed. |
| P1 | Per-device CUDA state landed; single-device CUDA decode remains bit-identical to pre-refactor; the existing DGX Spark / `make cuda-spark` and `make cuda-generic` builds still produce a working binary. |
| P2 | `layer_device[43]` plan abstraction lives in code with a deterministic contiguous default plus a CLI/env override; pretty-prints the plan at startup. |
| P3 | Multi-device tensor smoke (`tests/cuda_multi_device_smoke.c`) verifies per-device allocation and an HC-sized cross-device copy round-trips correctly. |
| P4 | A real `q2-imatrix` GGUF either loads across 8x32 GiB with per-device memory accounting, or fails with a recorded exact reason. This *is* the feasibility answer. |
| P5 | Greedy continuation A/B between CPU reference and 8-GPU CUDA on a short prompt is captured and graded; a SHIP / EXTEND / STOP verdict is filed. |
| P6 | `SPRINT-001-REPORT.md` + `SPRINT-001-FOLLOWUPS.md`; memory updated. |

---

## 3. Architecture

### 3.1 Hardware target

- Node: homelab `gpu-01`, 8x V100-SXM2-32GB (sm_70), no native FP8/MXFP4
  acceleration, NVLink-2 hybrid mesh between SXM2 pairs, PCIe between
  cross-pair traffic. (Per SPRINT-025 evaluation.)
- Visible to DS4 via `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`. The runtime must
  not assume the visible count is 8 — `cudaGetDeviceCount` is authoritative.
- Driver / CUDA version: whatever the homelab `homelab-k8s-dev` build image
  ships (CUDA 12.x); pinned in P0.

### 3.2 What the existing CUDA backend assumes (and why it must change)

`ds4_cuda.cu` is single-device today. The relevant globals are at file scope:

- `cudaSetDevice(0)` in `ds4_gpu_init` (`ds4_cuda.cu:1205-1222`).
- One `cublasHandle_t g_cublas` and one `g_cublas_ready` flag.
- One `g_model_host_base` / `g_model_device_base` model registration.
- One `g_model_ranges` weight cache vector, one `g_q8_f16_ranges` vector, one
  `g_cuda_tmp` scratch buffer, one `g_quality_mode` flag.
- The kernel call sites assume `cudaSetDevice` has already pointed at the right
  device, but nothing in the API surface ever sets a non-zero device.

A naive multi-GPU approach (call `cudaSetDevice(layer_device[il])` before each
kernel) would partially work for compute but corrupt the *caches*: a Q8→F16
preload pinned to device 0 would be touched as a device-1 pointer once the
layer ran there. The cleanest fix that preserves the single-device path is to
turn the file-scope globals into per-device arrays indexed by the current
device id, while keeping the API shape unchanged. (This is the same shape as
the per-device TmLib refactor in deepseek SPRINT-025 P2, but inside DS4.)

### 3.3 Device plan

A new struct in `ds4_gpu.h` (or a small internal header in `ds4_cuda.cu`) holds
the per-run plan:

```c
typedef struct ds4_gpu_plan {
    int      n_device;                 /* cudaGetDeviceCount, after filter */
    int      layer_device[DS4_N_LAYER];/* 0..n_device-1 per layer          */
    int      embed_device;             /* default layer_device[0]          */
    int      output_device;            /* default layer_device[DS4_N_LAYER-1] */
    int      peer_ok[8][8];            /* cached cudaDeviceCanAccessPeer    */
} ds4_gpu_plan;
```

Default placement is contiguous: 43 layers across N devices using
`base = 43 / N`, `extra = 43 % N`, so on 8 GPUs the first 5 devices get 6
layers each (0..29) and the last 3 devices get 5 layers each (30..42).
Embeddings on the layer-0 device, output head on the layer-42 device.

Override surface (env first; CLI flag added only if env proves awkward in
practice):

- `DS4_DEVICE_PLAN=auto` (default, contiguous as above).
- `DS4_DEVICE_PLAN=manual:0,0,0,0,0,0,1,1,...` (43 comma-separated device ids).
- `DS4_DEVICE_VISIBLE=0,1,2,3` (subset filter applied before plan generation;
  honors `CUDA_VISIBLE_DEVICES` if both are set, with `DS4_DEVICE_VISIBLE`
  intersecting that set).

The plan is generated once at `ds4_gpu_init`, validated against
`DS4_N_LAYER = 43` (intent constraint), and printed via `ds4_gpu_memory_report`
(extended; see §3.6).

### 3.4 Per-device CUDA state

`ds4_cuda.cu` grows a single `g_dev[8]` array (the constant `DS4_GPU_MAX_DEV`
defined as 8 — V100 SXM2 nodes top out at 8 visible GPUs; values beyond are
explicitly out of scope for this sprint):

```c
struct ds4_cuda_dev {
    int                    cuda_device;       /* logical -> physical map */
    int                    ready;
    cublasHandle_t         cublas;
    int                    cublas_ready;
    cudaStream_t           prefetch_stream;
    cudaStream_t           upload_stream;
    /* per-device weight caches */
    std::vector<cuda_model_range>     model_ranges;
    std::vector<cuda_model_arena>     model_arenas;
    std::unordered_map<uint64_t,size_t> model_range_by_offset;
    std::vector<cuda_q8_f16_range>    q8_f16_ranges;
    std::unordered_map<uint64_t,size_t> q8_f16_by_offset;
    std::vector<cuda_q8_f32_range>    q8_f32_ranges;
    std::unordered_map<uint64_t,size_t> q8_f32_by_offset;
    uint64_t                          model_range_bytes;
    uint64_t                          q8_f16_bytes;
    uint64_t                          q8_f32_bytes;
    void                            *tmp;
    uint64_t                          tmp_bytes;
    /* per-device flags */
    int                    quality_mode;
    int                    range_mapping_supported;
};
```

Wrap every `cudaSetDevice`/`cudaMalloc`/`cudaMemcpy` site with a small inline
helper `ds4_cuda_use(int dev)` that calls `cudaSetDevice` only if the current
device differs. The handful of cuBLAS call sites take a device argument; they
look up `g_dev[d].cublas` instead of the global. The model-cache helpers
(`cuda_model_range_ptr`, `cuda_q8_f16_cache_*`, `cuda_q8_f32_cache_*`) thread
the device through. (Mechanical change; the model load order in `ds4.c` decides
which device owns which range.)

**Single-device equivalence:** if `n_device == 1`, the behavior is byte-for-
byte identical to pre-refactor — the per-device arrays just have one populated
slot, and `ds4_cuda_use(0)` is a no-op. This is the kill gate at P1 (see §6).

### 3.5 Tensor ownership and the HC boundary

`ds4_gpu_tensor` today has three fields: `void *ptr; uint64_t bytes; int owner;`
(`ds4_cuda.cu:38`). It does **not** carry a device id. Adding one is the
smallest API change that unblocks layer sharding:

```c
struct ds4_gpu_tensor {
    void   *ptr;
    uint64_t bytes;
    int      owner;
    int      device;   /* NEW: cuda device this allocation belongs to */
};
```

The public `ds4_gpu.h` header does not need to change — `ds4_gpu_tensor` is
opaque. New allocation helpers (`ds4_gpu_tensor_alloc_on(int dev, ...)`) live
in the header but the existing `ds4_gpu_tensor_alloc` keeps its single-device
shape by routing to the current default device. The graph driver in `ds4.c`
opts in per-layer by switching to the new `_on` variants where needed.

The HC residual stream between layers is the cross-device boundary. Sizing
(intent constraint): `DS4_N_HC * DS4_N_EMBD * sizeof(float) = 4 * 4096 * 4 =
65,536 bytes` per layer transition, *plus* the optional split tensor used by
the fused HC-split path (small, < 1 KiB). On 8 GPUs with the §3.3 contiguous
plan, only 7 transitions are cross-device per token; the other 36 are within
one device. At 50 t/s the HC traffic ceiling is 7 * 64 KiB * 50 ≈ 22 MiB/s,
well inside even host-staged peer copies; NVLink-2 covers it with margin.

`ds4_gpu_tensor_copy` already takes a destination and a source tensor. With
device ids on the tensors, the implementation in `ds4_cuda.cu` picks one of
three paths:

1. `src.device == dst.device`: `cudaMemcpyAsync` D→D (current path).
2. `src.device != dst.device` and `peer_ok[src.device][dst.device]`:
   `cudaMemcpyPeerAsync`.
3. Otherwise: host-staged round trip through a pinned bounce buffer
   (`g_dev[src.device].tmp` writes to a small pinned host slab, then
   `cudaMemcpyAsync` H→D on the destination device). This is the conservative
   fallback when peer access fails (e.g., the V100 hybrid mesh has pairs that
   prefer PCIe).

Enable peer access at `ds4_gpu_init` for every supported `(i,j)` pair,
ignoring failures (they fall through to path 3). Cache the result in
`plan.peer_ok`.

### 3.6 Memory report

Extend `ds4_gpu_print_memory_report` (declared in `ds4_gpu.h`) so that on CUDA
builds it walks `g_dev[0..n_device-1]` and prints:

```
ds4: CUDA backend: 8 devices, plan=auto (DS4_N_LAYER=43)
ds4:  dev 0: Tesla V100-SXM2-32GB (sm_70)  total=32510 MiB free=32330 MiB
        layers=[0..5]  embed=yes  output=no
        weights=  0.0 MiB  q8_f16=  0.0 MiB  q8_f32=  0.0 MiB  scratch=  0.0 MiB
        peer: 0,1,2,3,4,5,6,7 -> 1,1,0,0,0,0,0,0
...
```

This is the device summary called out in the intent's Success Criteria. P4
re-runs it after the q2-imatrix load and the line above each `weights=` becomes
the actual budget evidence.

### 3.7 Model format decision (intent Q1, Q2, Q5)

This sprint commits to the **published `q2-imatrix` GGUF family** as the target
for Sprint 001. The reasoning:

- DS4's loader (`ds4.c:2283-2354`, `weights_validate_layout`) is already wired
  for it. `IQ2_XXS` gate/up + `Q2_K` down is the supported quant mix.
- The MXFP4/F8 path used by the deepseek-fork llama.cpp is **explicitly out
  of scope** for Sprint 001. Porting it would dwarf the multi-GPU work and
  defeat the bounded-spike framing in `SPRINT-025-V100-SPIKE-DIRECTION.md` and
  in the intent.
- The intent's Q2 ("Is the desired appliance target the published DS4 q2/q4
  GGUF family, the existing llama.cpp `DSv4-Flash-256e-fixed.gguf`, or both
  with a staged bridge?") is answered for this sprint as **just q2-imatrix**.
  A bridge to the FP4/FP8 model is filed in `SPRINT-001-FOLLOWUPS.md` as the
  most likely Sprint 002 work *if* this spike survives.
- The intent's Q5 ("Which existing DeepSeek kernels are worth importing first
  after fit") is answered **none in Sprint 001**. After fit, the prioritized
  list is filed in followups.

**Recorded in P0:** an explicit `gguf-tools/quality-testing` size check for the
local `q2-imatrix` file (~81 GiB), per-device budget at `43/8` layers, and a
dry-run estimate that the routed-expert weight ranges fit per device. The
existing `ds4_gpu_cache_model_range` already returns a clean error if a range
fails to register; the multi-device version surfaces that as a per-device fail
with the offending layer/tensor name.

### 3.8 Why no NCCL (intent Q4)

NCCL pays off when the per-token comm volume is large enough that ring/tree
reductions beat point-to-point. With the §3.5 budget (7 boundary copies of 64
KiB per token, no reductions in the decode hot path), `cudaMemcpyPeerAsync` is
sufficient and avoids the dependency, library install, and per-device-init
cost called out in deepseek SPRINT-025 P0.3. If a future sprint adds
row-sharded GEMMs or vocab-parallel LM-head, NCCL becomes interesting; it is
filed under followups.

### 3.9 Embedding and LM head placement (intent Q3)

- Token embeddings (`token_embd`, F16, 4096 × 129280 ≈ 1.0 GiB) stay on
  `embed_device` (= layer-0's device by default). Embedding lookup runs once
  per prefill chunk; the output is HC state for layer 0, which already lives
  on the same device.
- The LM head (`output`, Q8_0, 4096 × 129280 ≈ 528 MiB plus per-row scales)
  stays on `output_device` (= layer-42's device). The final HC state crosses
  from the last layer's device only when `output_device` differs from
  layer-42's device, which is **never** by default.
- Both choices are deliberate and minimal. Splitting the LM head across
  devices (vocab-TP) is a Sprint 002+ optimization — see deepseek
  `SPRINT-027-LM-HEAD-VOCAB-TP-SPIKE.md` for the alternative design space.

---

## 4. Implementation

### P0 — Build, device inventory, model-format dry run (1-2 days)

**Goal:** confirm DS4 builds for sm_70 and decide what GGUF we will actually
try to load.

1. **P0.1** — On the homelab `gpu-01` build env (microk8s pod or bare-metal as
   convenient), run `make clean && make cuda CUDA_ARCH=sm_70`. Capture stdout
   and stderr. Expected failure modes if any: unsupported intrinsics (e.g.,
   `__nv_bfloat16` paths, FP8 conversions). Each must be isolated to a small,
   documented patch or `#if __CUDA_ARCH__ >= 800` guard. **No new code in P0**
   beyond the smallest possible guards required to compile.
2. **P0.2** — `./tests/cuda_long_context_smoke` still passes on a single V100
   (`cudaSetDevice(0)`). This is the existing CUDA regression and is the
   baseline; if it regresses, P0 is failed.
3. **P0.3** — Add a one-shot `tools/cuda_device_inventory.c` (or a `--cuda-info`
   flag on `ds4`) that calls `cudaGetDeviceCount`, `cudaGetDeviceProperties`,
   and `cudaDeviceCanAccessPeer` for every pair, and prints the matrix used in
   §3.6. Commit the captured run as `docs/sprints/SPRINT-001-P0-inventory.md`.
4. **P0.4** — Stage `q2-imatrix` GGUF on the pod's shared storage (`/srv/dev/`
   per `homelab-k8s-dev` convention; per intent context the file is published
   at `huggingface.co/antirez/deepseek-v4-gguf`). SHA-256 check vs the
   published checksum. Record the file size (~81 GiB) and the bytes-per-layer
   estimate (routed gate/up `IQ2_XXS` + down `Q2_K` per layer).
5. **P0.5** — Confirm `weights_validate_layout` (`ds4.c:2285-2354`) accepts the
   file by mocking a load with `--cpu` first (no GPU residence). This is just a
   layout sanity gate; the CPU path will not generate but it will parse and
   validate.

**P0 kill gate (HARD):**

- `make cuda CUDA_ARCH=sm_70` produces a binary, **or** the failure list is
  bounded (≤ 5 distinct issues, each with a one-paragraph note) and `make
  cuda-spark`/`make cuda-generic` still build for the existing single-device
  use case. If sm_70 cannot build at all and the path forward is unclear, this
  sprint **STOPs** here.

### P1 — Per-device CUDA state (2-3 days)

**Goal:** lift `ds4_cuda.cu` globals into `g_dev[]`; preserve byte-identical
single-device behavior; expose the device count to callers.

1. **P1.1** — Add `DS4_GPU_MAX_DEV = 8` and `struct ds4_cuda_dev`; declare
   `g_dev[DS4_GPU_MAX_DEV]` and `g_n_device`. `ds4_gpu_init` calls
   `cudaGetDeviceCount`, fills `g_dev[i].cuda_device`, creates per-device
   cuBLAS handles, sets math mode per device, and prints the device summary.
2. **P1.2** — Introduce `ds4_cuda_use(int dev)`. Replace every
   `cudaSetDevice(0)` and bare `cudaMalloc` / `cudaMemcpy` with the device-
   aware versions. Audit by `grep -n cudaSet /Users/ravi/repos/ds4/ds4_cuda.cu`
   and `grep -n cublasCreate`; ~30 sites; mechanical.
3. **P1.3** — Add `device` to `struct ds4_gpu_tensor`; populate it in
   `ds4_gpu_tensor_alloc*` based on the current device at allocation time
   (default device 0 in single-device mode; `ds4_gpu_tensor_alloc_on(int dev,
   uint64_t bytes)` for callers that need explicit placement).
4. **P1.4** — Thread `device` through the model-cache helpers:
   `cuda_model_range_ptr`, `cuda_q8_f16_cache_*`, `cuda_q8_f32_cache_*`.
   Update the `unordered_map` keys to be `(offset, device)` so the same
   offset can be cached on multiple devices when (later) a tensor is mirrored.
   For Sprint 001, each offset is owned by exactly one device, but the type
   change is needed now.
5. **P1.5** — `ds4_gpu_print_memory_report` walks all devices, prints per-
   device totals + free + the cache breakdown from §3.6.
6. **P1.6** — Regression on single V100: build, run
   `./tests/cuda_long_context_smoke`, run `./ds4 --cuda -p "Hello"` against a
   small fixture if available. Byte-for-byte equivalence is verified by
   `--dump-logprobs /tmp/before.json` (pre-refactor) vs `/tmp/after.json`
   (post-refactor) on the same prompt with `--temp 0`.

**P1 kill gate:**

- ✅ Single-device CUDA `--dump-logprobs` output is byte-identical to the
  pre-refactor binary on a fixed 32-token greedy prompt.
- ✅ `./tests/cuda_long_context_smoke` passes.
- ✅ `make cuda-spark` / `make cuda-generic` still produce a working binary.
- ❌ If bit-equivalence fails, stop and bisect before continuing.

### P2 — Device plan abstraction (1-2 days)

**Goal:** `layer_device[DS4_N_LAYER]` exists, defaults to the §3.3 contiguous
plan, can be overridden, is printed at startup, and is consulted by exactly
the call sites that need to be device-aware.

1. **P2.1** — Add `ds4_gpu_plan g_plan` and the parser for `DS4_DEVICE_PLAN`
   and `DS4_DEVICE_VISIBLE`. `ds4_gpu_plan_init` fills the plan; validates
   that every entry is in `[0, n_device)`; refuses to start if the env value
   is malformed.
2. **P2.2** — A new public helper `ds4_gpu_layer_device(uint32_t il)` returns
   the device id for a layer. Embedding and output helpers expose
   `ds4_gpu_embed_device()` / `ds4_gpu_output_device()`.
3. **P2.3** — The graph driver in `ds4.c` *does not yet route to multiple
   devices*. P2 only lands the plan and prints it; the actual per-layer
   `ds4_cuda_use(layer_device[il])` switch is in P3.
4. **P2.4** — Plan printout is part of `ds4_gpu_print_memory_report` (§3.6).
5. **P2.5** — One-line summary in `--cuda-info` output: device count, plan
   mode, embed/output device, peer-access matrix density (e.g.,
   `peer: 8/8 directs, 56/56 pairs`).

**P2 gate:**

- ✅ `DS4_DEVICE_PLAN=auto` on 1 GPU yields the existing single-device behavior.
- ✅ `DS4_DEVICE_PLAN=auto` on 8 GPUs yields layers [0..5] [6..11] ...
  [36..42] over devices 0..7 (5 devices get 6 layers, 3 get 5; assignment
  exact and documented).
- ✅ `DS4_DEVICE_PLAN=manual:...` rejects malformed input with a clear error.

### P3 — Multi-device smoke (2-3 days)

**Goal:** Sprint 001's correctness scaffolding — proves that the per-device
state and the HC-sized cross-device copy actually work, before we attempt a
real model load.

1. **P3.1** — Add `tests/cuda_multi_device_smoke.c` (modeled on
   `tests/cuda_long_context_smoke.c`). It:
   - calls `ds4_gpu_init`;
   - skips with success if `n_device < 2`;
   - allocates an HC-sized tensor (64 KiB) on each visible device;
   - writes a unique pattern to each;
   - calls `ds4_gpu_tensor_copy` from device i to device j for every pair
     `(i,j)`, including `(i,i)`, and verifies the destination has the source
     pattern;
   - tears down cleanly.
2. **P3.2** — Wire it into the Linux build as `make cuda-regression` extension:
   the existing target `tests/cuda_long_context_smoke` keeps running, and the
   new test is added behind it (it short-circuits as a pass when only one GPU
   is visible, so the existing DGX Spark CI path is unaffected).
3. **P3.3** — Capture the elapsed time per pair (it should be O(10 µs) per
   64-KiB peer copy on NVLink, O(50-200 µs) on host-staged fallback). Include
   in the test stderr line for later spot-checking.
4. **P3.4** — A second sub-test exercises *concurrent* allocations + copies:
   allocate 16 MiB scratch on device 0 and device 1 simultaneously, run two
   parallel copy loops, sync, verify no `cudaErrorInvalidContext`. This
   catches the most common per-device-state leak (a kernel launched on the
   wrong context because `cudaSetDevice` was not re-set after a callback).
5. **P3.5** — Print the per-device memory report at end of test.

**P3 gate:**

- ✅ On 1 GPU: smoke is `skip (pass)`; long-context smoke continues to pass.
- ✅ On 8 GPUs: every `(i,j)` pair round-trips an HC tensor with bit-equal
  contents.
- ✅ Concurrent sub-test reports no CUDA error.
- ❌ If any pair fails, stop. Likely cause is either the per-device state
  refactor (P1) or peer-access misconfiguration; bisect before P4.

### P4 — q2-imatrix multi-device load (2-3 days)

**Goal:** the published `q2-imatrix` GGUF actually resides across 8 V100s with
the plan from §3.3. *This is the feasibility answer the intent is asking for.*

1. **P4.1** — Extend the model load path in `ds4.c` to consult
   `ds4_gpu_layer_device(il)` for every per-layer weight tensor. The
   non-per-layer tensors (`token_embd`, `output`, output norm/HC) go to
   `embed_device` and `output_device` respectively.
2. **P4.2** — Per-device `ds4_gpu_cache_model_range` and
   `ds4_gpu_cache_q8_f16_range` calls; check for OOM at every step. On OOM,
   print the device id, the offending tensor name, the current per-device
   usage, and stop with a non-zero exit. Do not silently fall through to host-
   mapped weights — that path is known to be unworkable on V100 for the
   routed expert ranges and will be detected as a separate Sprint 002+
   followup.
3. **P4.3** — Run with `q2-imatrix` from P0.4. Capture the post-load memory
   report. Expected per-device usage at default plan: `81 GiB / 8 ≈ 10.1
   GiB` of routed-expert weights + ~0.5 GiB of dense weights + 1 GiB of token
   embed on device 0 + 0.5 GiB of output head on device 7 + ~2 GiB of KV at
   8 K context + ~1 GiB of cuBLAS / scratch + ~1 GiB of driver overhead. ≈
   13-16 GiB on most devices, ≈ 17-18 GiB on devices 0 and 7. Headroom on
   32 GiB cards: ≥ 14 GiB per device.
4. **P4.4** — If the default plan is imbalanced (a device exceeds 22 GiB or
   a device sits below 8 GiB), try `DS4_DEVICE_PLAN=manual:...` to rebalance
   and record the working assignment in the report.
5. **P4.5** — Confirm the existing per-device Q8→F16 cache budget (controlled
   by `DS4_CUDA_Q8_F16_CACHE_MB` / `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB`) reserves
   enough headroom; the existing 5%-of-total or 4 GiB minimum reserve
   (`ds4_cuda.cu:357-373`) translates to ~1.6 GiB on a 32 GiB V100, which is
   reasonable. Document the override values if used.

**P4 kill gate (HARD — this is the feasibility decision):**

- ✅ `q2-imatrix` fits in 8x V100-SXM2-32GB with the contiguous plan, every
  per-device usage ≤ 28 GiB at 8 K ctx FP16 KV.
- ✅ OR fits at 4 K ctx with FP16 KV.
- ❌ If fits requires either > 28 GiB on any device, or context < 2 K, or
  forces dropping routed-expert ranges to host-mapped pages, the sprint
  **STOPs** with a recorded report. The next step would be either a smaller
  quant (`q2` legacy non-imatrix, smaller routed mix), or model-format work
  (Sprint 002 FP4/FP8 bridge), neither of which is in scope here.

### P5 — Coherence smoke + verdict (1-2 days)

**Goal:** one short greedy continuation across 8 GPUs, compared against the
existing CPU reference for the same input. Not a benchmark.

1. **P5.1** — `ds4_gpu_layer_device` is consulted in the graph driver's
   per-layer dispatch (`ds4.c` decode/prefill loop): switch device, run
   attention/FFN, copy HC state to the next layer's device.
2. **P5.2** — Run `./ds4 --cuda -p "The capital of France is" --temp 0
   --tokens 32 --dump-logprobs /tmp/v100_8gpu.json`.
3. **P5.3** — Run the same input on `./ds4 --cpu` (the diagnostic path) and
   capture `/tmp/cpu.json`. Compare leading-token match; ≥ 30/32 tokens
   matching is a soft pass (CPU and CUDA quants/dequant orderings differ
   enough that exact match is not guaranteed for IQ2_XXS/Q2_K, and bit-equal
   logits across paths is not the intended gate for the published DS4
   GGUFs — `tests/test-vectors` is the official correctness check).
4. **P5.4** — Try `./ds4_test --logprob-vectors` against any captured
   official-vector that is small enough to fit the 8x V100 stack. If the
   official continuation passes, that is much stronger than the CPU eyeball;
   if it does not yet pass, the sprint still ships with an EXTEND verdict and
   a Sprint 002 followup to chase the discrepancy.
5. **P5.5** — Write `docs/sprints/SPRINT-001-REPORT.md`. Required sections:
   - device inventory dump (`SPRINT-001-P0-inventory.md` reference);
   - per-device memory after `q2-imatrix` load (from §3.6);
   - the actual `layer_device[]` used;
   - peer-access matrix;
   - the 32-token sample(s) from P5.2/P5.3 with the leading-token match;
   - the verdict: **SHIP** (coherent + fits), **EXTEND** (fits but coherence
     gap, needs more work), or **STOP** (any P0–P4 kill gate fired).

**P5 gate:**

- ✅ A 32-token greedy continuation completes without crash.
- ✅ Output is non-empty, non-`#`-loop, parses as UTF-8.
- ✅ Verdict is recorded.

### P6 — Close-out (0.5 day)

1. `docs/sprints/SPRINT-001-FOLLOWUPS.md` — prioritized list. Likely entries
   (per intent Q5 and SPRINT-025 evaluation):
   - FP4/FP8/MXFP4 loader bridge (Sprint 002 candidate).
   - Importing the TurboMind MXFP4 routed-expert sm70 path
     (`/Users/ravi/repos/deepseek/ggml/vendor/turbomind`) after the format
     bridge.
   - `tools/tc-grid/v13_rf_v6` dense kernel evaluation for shared experts.
   - NCCL evaluation only if Sprint 002 introduces a row-sharded path.
   - Vocab-TP LM head (deepseek `SPRINT-027-LM-HEAD-VOCAB-TP-SPIKE.md`).
   - Multi-slot decode and request batching.
2. Memory: append `private-fork v100 feasibility (2026-05-..) — verdict X`
   to the project memory.
3. Tag `sprint-001-close` on the branch.
4. No upstream PR (per AGENT.md / private-fork stance).

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `ds4_cuda.cu` | Per-device `g_dev[]` array; replace `cudaSetDevice(0)` and `g_cublas` globals with per-device state; thread `device` through model-cache helpers; extend `ds4_gpu_print_memory_report`. |
| `ds4_gpu.h` | Opaque `ds4_gpu_tensor` carries a device id (definition stays internal); add `ds4_gpu_tensor_alloc_on(int dev, uint64_t bytes)`; add `ds4_gpu_layer_device`, `ds4_gpu_embed_device`, `ds4_gpu_output_device`, `ds4_gpu_device_count` (all return 0/1 sensibly on single-device builds). |
| `ds4.c` | Model load consults `ds4_gpu_layer_device`; embed/output placement via `ds4_gpu_embed_device`/`ds4_gpu_output_device`; per-layer dispatch switches device. |
| `ds4_cli.c` | New `--cuda-info` flag for the P0.3 device dump; `--cuda` path prints the plan. |
| `Makefile` | `make cuda-regression` runs both `tests/cuda_long_context_smoke` and `tests/cuda_multi_device_smoke`. |

### New

| Path | Purpose |
|---|---|
| `tests/cuda_multi_device_smoke.c` | Per-device allocation + HC-sized cross-device copy round trip. |
| `docs/sprints/SPRINT-001-P0-inventory.md` | Captured `nvidia-smi -L` + `--cuda-info` output from the homelab pod. |
| `docs/sprints/SPRINT-001-REPORT.md` | Headline measurement + ship/extend/stop verdict. |
| `docs/sprints/SPRINT-001-FOLLOWUPS.md` | Sprint 002+ candidates (model-format bridge, kernel imports, NCCL, vocab-TP). |

### Unmodified by design (flagged for caution)

| Path | Note |
|---|---|
| `ds4_metal.m`, `metal/*.metal` | Metal backend stays as-is. Multi-device is a CUDA-only concern this sprint. |
| `ds4_server.c`, `ds4_bench.c`, `ds4_eval.c` | API surface unchanged. The server's single-graph-worker model continues to work with the multi-device backend. |
| `gguf-tools/*` | Untouched. |
| Any file in `metal/`, `dir-steering/`, `speed-bench/`, `tests/test-vectors/` | Untouched. |

### Possibly modified (flagged for caution; only if forced by P0)

| Path | Note |
|---|---|
| `ds4_iq2_tables_cuda.inc` | Only if a sm70 incompat is found in the IQ2 tables expansion. Unlikely. |
| `ds4_cuda.cu` near IQ2/Q2_K dequant kernels | Only if `__nv_bfloat16` or FP8 conversion intrinsics block sm70 compile and a minimal `#if __CUDA_ARCH__ >= 800` guard is required. The guarded code path must be unreachable on sm_70 (sm_70 cannot reach it anyway via the existing supported tensor types). |

---

## 6. Definition of Done

A *bounded, kill-gated* DoD. Each item is a hard precondition for advancing to
the next phase; missing any of them is a STOP signal, not a deferral.

1. ✅ `make cuda CUDA_ARCH=sm_70` produces a working binary on the homelab
   `gpu-01` build env, **or** the compile failure list is bounded to ≤ 5
   documented issues with explicit guards/workarounds.
2. ✅ `--cuda-info` prints `cudaGetDeviceCount`, per-device capability/memory,
   and the peer-access matrix. Output committed as
   `SPRINT-001-P0-inventory.md`.
3. ✅ `ds4_cuda.cu` carries no file-scope `g_cublas` / `g_model_*` / `g_q8_*`
   / `g_cuda_tmp` globals; every previously-global field lives in
   `g_dev[DS4_GPU_MAX_DEV]` indexed by device. `grep '^static [^s].*= 0' ds4_cuda.cu`
   shows no surviving single-device caches at file scope.
4. ✅ On a single V100, `./ds4 --cuda --temp 0 --tokens 32 --dump-logprobs`
   output is byte-identical to the pre-refactor binary on a fixed prompt.
5. ✅ `make cuda-spark` and `make cuda-generic` still produce working
   binaries; existing DGX Spark / generic-CUDA users are unaffected.
6. ✅ `ds4_gpu_layer_device(il)` returns a device id from 0..n_device-1 for
   every `il ∈ [0, 43)`. `DS4_DEVICE_PLAN=auto` and
   `DS4_DEVICE_PLAN=manual:...` both round-trip through the parser.
7. ✅ `tests/cuda_multi_device_smoke` passes on 8 V100s: every device-pair
   HC-sized copy round-trips bit-equal contents; concurrent allocation and
   copy generates no CUDA errors.
8. ✅ Published `q2-imatrix` GGUF (SHA-256 verified) loads across 8 V100s with
   the `auto` plan; every device's post-load usage ≤ 28 GiB at 8 K ctx FP16
   KV (or ≤ 28 GiB at 4 K ctx if 8 K fails).
9. ✅ A 32-token greedy continuation completes without crash, produces UTF-8,
   passes an eyeball coherence check.
10. ✅ `SPRINT-001-REPORT.md` records the device inventory, post-load memory,
    `layer_device[]`, peer-access matrix, sample output, and a SHIP / EXTEND
    / STOP verdict.
11. ✅ `SPRINT-001-FOLLOWUPS.md` is filed.
12. ✅ Tag `sprint-001-close` exists on the branch.

**Kill gates (re-stated, by phase):**

- **P0** — sm_70 build is impossible and unbounded → STOP.
- **P1** — single-device bit-equivalence regresses → STOP.
- **P3** — any device-pair HC copy fails → STOP.
- **P4** — `q2-imatrix` does not fit in 8x32 GiB at ≥ 2 K ctx → STOP.
- **P5** — greedy decode crashes or produces structural gibberish (e.g.,
  `# # #` loop, all-NaN logits) → STOP.

Any STOP at any phase still ships P0–(failing-phase-1)'s landed work and the
report; the sprint does not roll back the per-device refactor in P1 even if
P4 or P5 fails. The single-device path is preserved either way.

---

## 7. Risks & Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | sm_70 build fails because some kernel uses `__nv_bfloat16`, FP8 conversions, or other sm_80+-only intrinsics | Medium | High | P0.1 captures the failure list; `#if __CUDA_ARCH__ >= 800` guards added only for unreachable-on-sm70 code paths; if a critical path is sm_80-only, this is the STOP signal at P0 |
| 2 | Per-device refactor breaks single-device bit-equivalence (P1) | Medium | High | P1.6 dumps `--dump-logprobs` before and after; bisect if mismatch; do not advance to P2 until equivalence is restored. The repo already has `tests/cuda_long_context_smoke` as a second guard. |
| 3 | V100 SXM2 peer access fails for some pairs (hybrid mesh) | High | Low | §3.5 path 3 (host-staged via pinned bounce) is the fallback; the HC payload is 64 KiB so even host-staged copies are well under hot-path budgets. P3.1 verifies every pair. |
| 4 | `q2-imatrix` does not fit due to imbalanced routed-expert ranges | Medium | High | P4.4 hand-tunes `DS4_DEVICE_PLAN=manual:...`; if still no fit, this is the STOP signal at P4. |
| 5 | `cudaHostRegister`/HMM/range-mapping behavior on V100 differs from DGX Spark and breaks the existing model-cache path | Medium | Medium | The existing `range_mapping_supported = 1` flag falls through to plain `cudaMalloc + cudaMemcpy` if `cudaHostRegister` fails (`ds4_cuda.cu:262-275`); per-device flag means a per-device fallback is automatic. |
| 6 | Cross-device pointer used by a kernel after `cudaSetDevice` is not re-set after an asynchronous host callback | Medium | High | `ds4_cuda_use(int dev)` is the *only* device-switch path; the helper records the requested device and re-applies it on every entry. P3.4 concurrent-sub-test guards against this regressing. |
| 7 | The graph driver in `ds4.c` accidentally accesses a tensor through a stale device pointer (e.g., a scratch tensor allocated on the wrong device) | Medium | High | The `device` field on `ds4_gpu_tensor` is checked against the current device in `ds4_gpu_tensor_copy`; on mismatch, log the device pair and route to peer/host-staged path; never silently `cudaMemcpyAsync` a foreign-device pointer. |
| 8 | OOM on device 0 because embeddings + layer-0 weights + scratch exceed budget | Low-Medium | High | §3.9 keeps embeddings on the layer-0 device; if device-0 budget is tight, P4.4 plan-tune moves layer 0 to device 1 so device 0 only holds embeddings + KV. |
| 9 | The intent's stop-loss (1 week) is reached before P4 completes | Medium | Medium | P0–P3 each have a 1-2 day cap; if P0 stretches past day 3, the sprint stops with whatever landed (P0/P1 alone are still useful primitives for a future spike). |
| 10 | `q2-imatrix` quality is degraded enough on the 8-GPU layout to make the coherence smoke ambiguous (intent uncertainty: high) | Low | Medium | P5.4 falls back to the `tests/test-vectors` regression for an external ground truth; if both eyeball and vector test disagree, the verdict is EXTEND, not SHIP, and the gap is filed as a Sprint 002 followup. |
| 11 | Per-device cuBLAS handle creation order produces a different math-mode default on some devices (TF32 vs default) and decode drifts | Low | Medium | `ds4_cuda_dev::quality_mode` carries the same setting per device; P1.5 prints math mode per device in the memory report; P5.4 catches drift if any. |
| 12 | The intent's open question 6 stop-loss ("one week without coherent q2 output") is reached but the team is invested | Low | Medium | The kill gates in §6 are HARD; even if work feels close, a STOP at P4 means the verdict is STOP. Followups capture what to attempt next. |

---

## 8. Security Considerations

- **No network surface.** This sprint does not change the server; the existing
  `ds4-server` still serializes through one graph worker (per
  `ds4_server.c:4-11`). The HTTP API is unchanged. Multi-GPU does not expose
  any new socket.
- **Model file trust.** The `q2-imatrix` GGUF is downloaded from
  `huggingface.co/antirez/deepseek-v4-gguf` and SHA-256 verified at P0.4. The
  `weights_validate_layout` path (`ds4.c:2285-2354`) is the existing structural
  trust boundary; this sprint does not relax it.
- **No new dependencies.** No NCCL, no MPI, no UCX. Only CUDA + cuBLAS as
  today.
- **Process privilege.** The runtime continues to use `cudaHostRegister`
  (requires the standard CUDA driver permissions, which the homelab pod already
  has); no new capabilities are needed.
- **Disk KV cache.** Unchanged; per-device weight caches are *not* the same as
  the disk KV cache, which is a separate `read`/`write` path under
  `--kv-disk-dir`.
- **Multi-GPU pointer hygiene.** A pointer allocated on device A and
  dereferenced on device B is the classic CUDA security/safety hazard. §3.5
  path 3 (host-staged) and the `device` field on `ds4_gpu_tensor` enforce a
  hard check: cross-device pointer use without an explicit copy is a fatal
  error logged with both device ids. This is enforced uniformly so no future
  patch can silently regress it.
- **No outbound telemetry.** The memory report and `--cuda-info` write only to
  stderr / report files; the dotfiles repo memory note is local.
- **AI-generated code disclosure.** Per repo norms, all code in this sprint
  is AI-assisted; the fork is private (`rapatel0/ds4`), no upstream PR, and
  no AI-attributed contributions to `antirez/ds4`.

---

## 9. Dependencies

1. **Hardware.** Homelab `gpu-01` node, 8x V100-SXM2-32GB, CUDA driver
   compatible with the CUDA toolkit version on the build image. Existing
   single-GPU usage of this node continues; coordinate so we don't fight a
   running llamacpp deepseek pod for VRAM. (Cross-checked with the
   `homelab-k8s-dev` skill.)
2. **CUDA toolkit.** Whatever the homelab build image ships (CUDA 12.x).
   Pinned in P0.1 commit message.
3. **GGUF.** Published `q2-imatrix` from `huggingface.co/antirez/deepseek-v4-
   gguf`, ~81 GiB. Staged on cluster-local storage before P3.
4. **DS4 branch.** Current `main` (`ef0a490 Add ds4-server working-directory
   option`); no rebase against `upstream antirez/ds4` required for this
   sprint. Origin is `rapatel0/ds4`.
5. **Existing tests.** `make cuda-regression` (which currently runs
   `tests/cuda_long_context_smoke`) is the baseline; this sprint extends it,
   does not replace it.
6. **Reference repo (read-only).** `/Users/ravi/repos/deepseek` for:
   - `docs/sprints/SPRINT-025-DS4-EVAL.md` (the constraints recap).
   - `docs/sprints/SPRINT-025-V100-SPIKE-DIRECTION.md` (the recommended
     bounded-spike framing this sprint implements).
   - `docs/sprints/SPRINT-025-TURBOMIND-HANDOFF.md` (MXFP4 fix; relevant only
     to Sprint 002+ format-bridge followup).
   - `tools/tc-grid/*`, `ggml/vendor/turbomind/*` — out of scope here, listed
     for Sprint 002+ kernel-import sequencing.
7. **Not a dependency.** NCCL, MPI, UCX, the deepseek-fork llama.cpp build
   image, `nccl-tests`, NSYS profiling, `ds4_test --logprob-vectors` against a
   long-context vector. `--logprob-vectors` is *optional* in P5.4 if it
   happens to be tractable.

---

## 10. Open Questions

The intent listed six. Sprint 001 commits to answers for four of them; the
remaining two are explicitly punted to Sprint 002+ followups.

1. **Intent Q1: Should Sprint 001 require loading the published antirez
   q2-imatrix GGUF across 8 GPUs, or is a smaller multi-GPU skeleton plus exact
   format-failure report enough?** — **Answered: full q2-imatrix load is the
   feasibility gate at P4.** A skeleton without a real fit is not feasibility.
2. **Intent Q2: Is the appliance target the published DS4 q2/q4 family, the
   llama.cpp `DSv4-Flash-256e-fixed.gguf`, or both with a staged bridge?** —
   **Answered for Sprint 001: q2-imatrix only.** Bridge to FP4/FP8/MXFP4 is
   the top Sprint 002 candidate, filed in followups.
3. **Intent Q3: Should the first multi-GPU implementation shard only layer
   weights and KV by layer, leaving embeddings on device 0 and output head on
   the last device?** — **Answered: yes** (§3.9). Vocab-TP LM head is a
   separate, larger sprint.
4. **Intent Q4: Is NCCL required for the homelab V100 stack, or is 64 KiB HC
   state transfer via peer/host `cudaMemcpyPeerAsync` sufficient for Sprint
   001?** — **Answered: peer/host is sufficient** (§3.8). NCCL is filed for
   when a future sprint adds reduction-bearing operations.
5. **Intent Q5: Which existing DeepSeek kernels are worth importing first
   after fit?** — **Punted to followups.** P5/P6 produces a prioritized list,
   most likely the TurboMind MXFP4 routed-expert path (after the format
   bridge from Q2), then `tc-grid v13_rf_v6` for dense shared-expert paths.
6. **Intent Q6: What is the stop-loss threshold for this fork?** —
   **Answered: §6 hard kill gates + one calendar week from P0 start.**
   Whichever fires first. The "fail to outperform / materially simplify the
   llama.cpp path" threshold is not a Sprint 001 gate (there is no perf
   target this sprint); it is the gate that decides whether Sprint 002
   continues, and it lives in `SPRINT-001-FOLLOWUPS.md`.

New questions surfaced by this draft that need answers during execution:

- **Q-A: Should `DS4_DEVICE_PLAN` be promoted to a CLI flag in `ds4_cli.c`?**
  Default is env-only for Sprint 001; if operator UX is awkward, promote in
  P6. Not a gate.
- **Q-B: Is the existing `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` heuristic correct
  on 32 GiB cards?** The current min-reserve is 4 GiB or 5 %; on a V100 that
  rounds to 4 GiB which may be over-conservative. Tune if P4 finds spare
  budget; not a gate.
- **Q-C: How do we handle the V100 pod sharing `gpu-01` with the existing
  llamacpp deepseek workload?** Coordinate before P4 (stop or reschedule the
  competing pod); the multi-GPU smoke in P3 can run alongside it, but the q2-
  imatrix load cannot.

---

## 11. Outcome Contract

Sprint 001 ships if and only if **all** of the following are true at sprint
close:

- `make cuda CUDA_ARCH=sm_70` builds the four DS4 binaries cleanly.
- Single-device CUDA decode is bit-identical to the pre-refactor binary.
- `q2-imatrix` resides across 8x V100-SXM2-32GB with the recorded plan and per-
  device memory accounting within budget.
- A greedy 32-token continuation produces non-empty, UTF-8, eyeball-coherent
  output on the 8-GPU stack.

If any of those is false, the sprint files a **STOP** verdict and ships the
partial work (per-device refactor + multi-device smoke at minimum). The
reusable primitive — per-device CUDA state + device plan abstraction — lands
regardless.

The TPS number in the report is **not** a gate; it is the baseline for any
future Sprint 002 performance work.
