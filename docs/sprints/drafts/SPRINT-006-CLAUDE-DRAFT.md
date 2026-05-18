# SPRINT-006 — Multi-GPU Execution Context And Layer Skeleton (CLAUDE draft)

**Status:** DRAFT 2026-05-17
**Predecessor:** SPRINT-005 (first resident BF16 gather/expand probe)
**Successor:** Single-slot decode correctness, gated on this sprint shipping a typed-descriptor context, an HC relay primitive, and an execution-format policy that is enforced rather than aspirational.

---

## Overview

Sprints 001-004 built the offline-to-resident pipeline: every source tensor is
inventoried, manifested, packed into per-GPU shards, reconciled, and uploaded
into eight CUDA device arenas with reserve to spare. Sprint 005 then proved
that those resident bytes can be addressed and converted dtype-correctly: a
BF16 row-gather/expand probe walks `token_embd.weight` from `ds4_gpu_arena`
memory on GPU 0 and produces bit-exact F32 samples.

The next missing piece is a runtime shape. Today the CUDA code in
`ds4_cuda.cu` is global and single-device: `g_cublas`, `g_cuda_tmp`,
`g_model_host_base`, `g_model_device_base`, `g_model_ranges`,
`g_model_arenas`, and the prefetch/upload streams all assume one current
device and one model map. The pack-arena sidecar is bolted on next to those
globals. Decode cannot proceed against this shape because there is nowhere to
hang per-GPU streams, handles, scratch, descriptors, or relay buffers — and
no policy file that says which V100-native compute format each tensor family
is allowed to feed.

Sprint 006 fills that gap **without enabling decode**. It introduces a typed,
opaque, eight-way `ds4_v100_context` that owns per-GPU streams, cuBLAS
handles, scratch, descriptor tables, and HC relay buffers; it loads typed
resident tensor descriptors from the pack index for at least the global BF16
embedding and one representative layer-owned tensor span; it adds a minimal
HC relay primitive sized for `[active_slots][4][4096]` boundary payloads and
proves device-to-device transfer (or pinned-host fallback) on the V100 pod;
and it adds a layer skeleton walker that visits the planned 8-stage layer map
and validates ownership, descriptor presence, and boundary allocations
without launching attention or MoE math.

Critically, V100 has **no native BF16, FP8, or FP4 tensor-core execution**.
Sprint 005's correction makes that explicit and Sprint 006 must encode it as
an enforced policy: BF16 source bytes are diagnostic-only or expanded to
FP16/F32 at the kernel boundary; FP8 and MXFP4 source packs feed
unpack-then-FP16-HMMA kernels; activations are FP16; control/reduction is
FP32. The context must classify each tensor family against this policy and
**fail closed** when a caller asks for an unsupported pairing — for example,
"do BF16 GEMM on the V100." This sprint does not write any new tensor-core
kernels. It writes the contract that the next sprint's kernels will register
against.

**Outcome contract:**

- **SHIP** if: `ds4_v100_context` exists as an opaque 8-GPU context with
  per-GPU streams, cuBLAS handles, scratch arenas, weight-arena references,
  relay buffers, and typed resident tensor descriptors; the context can be
  initialized from a real pack index (or a metadata-only subset thereof);
  the execution-format policy is encoded in code and emitted in a context
  report; an HC relay primitive transfers a `[active_slots][4][4096]` FP16
  payload between GPU 0 and GPU 1 on the V100 pod (or falls back to pinned
  host with the same shape); the layer skeleton walks the 8-stage map and
  validates descriptors for at least one full layer's tensor family list;
  local model-less tests cover policy classification, descriptor bounds,
  layer-map validation, and invalid configs; CUDA synthetic tests pass on
  the V100 pod; source-model generation remains guarded.
- **EXTEND** if: local context init, descriptor build, policy classifier,
  layer skeleton walker, and stub relay land, but the cluster CUDA HC relay
  smoke or full real-pack context init is blocked by infrastructure or pod
  availability. The blocker and the exact missing validation are recorded.
- **STOP** if: a typed context cannot coexist with the existing CUDA globals
  without a destabilizing rewrite that pulls in decode; or the planned
  per-GPU scratch + relay footprint plus existing arena residency drives any
  GPU below the declared reserve.

---

## Use Cases

Each phase produces a useful artifact even if a later phase slips:

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | Local build remains green; Sprint 005 artifacts remain valid; Sprint 006 entry point in `Makefile` exists. |
| P1 | `ds4_v100_context.h` defines the typed, opaque context, descriptor structs, the execution-format policy enum, and the kernel-family classifier. Header documents the V100 ground rules (no native BF16/FP8/FP4). |
| P2 | The host/stub context implementation can be opened from a pack index, builds the per-GPU descriptor tables, classifies each tensor family against policy, and prints a context report. Layer skeleton walker validates descriptor presence per layer. |
| P3 | CUDA implementation initializes per-GPU streams, cuBLAS handles, scratch, and relay buffers on a real V100 (single- or multi-GPU node). Reports `cudaMemGetInfo`, P2P matrix, residency class, and reserve headroom. |
| P4 | HC relay primitive transfers `[active_slots][4][4096]` FP16 payloads device-to-device (or pinned-host fallback) between two GPUs, with byte equality verified after relay. |
| P5 | Real-pack context smoke initializes the typed context against the persistent pack on the V100 pod, walks the 8-stage layer map, and emits a topology/memory/policy report. |
| P6 | Sprint report enumerates the API contract Sprint 007 can target, the policy table the kernel sprints can register against, and the deferred work. |

---

## Architecture

### Source Of Truth

`docs/architecture/DS4-V100-LAYOUT.md` remains the architecture anchor for
the 8-stage layer map, per-GPU arena composition, source-vs-runtime dtype
tables, and `[active_slots][4][4096]` HC boundary payload shape. Sprint 006
implements the runtime that consumes that document; it does not modify the
layout. If the policy classifier disagrees with a row in the layout table,
that is a STOP that escalates to a layout review, not a license to silently
reclassify.

### Module Boundaries

```text
ds4_v100_context.h / ds4_v100_context.c
    opaque per-GPU context, descriptor tables,
    execution-format policy enum + classifier,
    layer-map walker, relay-buffer accessors,
    fail-closed kernel-family resolver.

ds4_v100_context_cuda.cu
    CUDA-side context state:
    per-GPU streams, cuBLAS handles, scratch,
    HC relay device buffers, P2P enable,
    pinned-host fallback path.

ds4_gpu.h, ds4_cuda.cu, ds4_gpu_arena_stub.c
    unchanged behavior; the context borrows
    existing arenas through ds4_gpu_arena_*.

ds4_pack.h, ds4_pack.c
    unchanged.  The context reads from the
    existing pack-entry API to populate descriptors.

tools/ds4-v100-context-smoke.c
    standalone diagnostic that opens the context,
    walks the layer skeleton, prints the report,
    and (when at least two GPUs are visible) runs
    the HC relay primitive.

tests/v100_context_smoke.c
    model-less context, policy, descriptor,
    and layer-map tests.

tests/cuda_v100_context_smoke.c
    direct-CUDA synthetic context, scratch, and
    relay-buffer tests.

tests/cuda_hc_relay_smoke.c
    direct-CUDA HC relay primitive test (requires
    >=2 visible GPUs; otherwise skipped with a
    recorded reason).
```

The new code is a **sidecar** to the existing global CUDA state, identical in
spirit to the Sprint 004 arena sidecar. The legacy `g_cublas`,
`g_cuda_tmp`, `g_model_*`, and `cuda_model_arena` paths are not modified.
`ds4.c`'s engine open is **not wired** to the new context in this sprint
beyond an optional inspect-only path; production decode wiring is the next
sprint's problem.

### Opaque Context Shape

```c
typedef struct ds4_v100_context ds4_v100_context;
typedef struct ds4_v100_gpu_context ds4_v100_gpu_context;
typedef struct ds4_v100_tensor_desc ds4_v100_tensor_desc;

typedef enum {
    DS4_V100_INIT_PROBE_ONLY = 0,   /* descriptors + scratch + relay; arenas optional */
    DS4_V100_INIT_USE_EXISTING_ARENAS = 1, /* reference Sprint 004 arenas without uploading */
    DS4_V100_INIT_FULL_RESIDENT = 2  /* full upload via existing arena API */
} ds4_v100_init_mode;

typedef struct {
    const char *pack_index_path;
    const char *shard_dir;          /* may be NULL when only gguf provider used */
    const char *source_model_path;  /* GGUF; required for metadata */
    int         expected_gpus;      /* 8 for production; 1+ allowed for tests */
    uint64_t    scratch_bytes_per_gpu;
    uint64_t    relay_max_active_slots;
    uint64_t    reserve_mib;
    ds4_v100_init_mode mode;
} ds4_v100_context_options;

int  ds4_v100_context_open(ds4_v100_context **out,
                           const ds4_v100_context_options *opts,
                           char *err, size_t errlen);
void ds4_v100_context_close(ds4_v100_context *ctx);

int  ds4_v100_context_n_gpus(const ds4_v100_context *ctx);
const ds4_v100_gpu_context *
     ds4_v100_context_gpu(const ds4_v100_context *ctx, int gpu);
void ds4_v100_context_print_report(const ds4_v100_context *ctx, FILE *fp);
```

`ds4_v100_gpu_context` is opaque to callers. Internally it owns:

- the device ordinal, PCI bus ID, and visible-index pair (so
  `CUDA_VISIBLE_DEVICES` remappings are interpretable);
- a borrowed pointer to the GPU's `ds4_gpu_arena` (the weight arena), or
  NULL if `mode == DS4_V100_INIT_PROBE_ONLY` and no upload happened;
- one default execution stream and one relay stream;
- one cuBLAS handle bound to the default stream;
- a scratch device allocation of size `scratch_bytes_per_gpu`;
- two relay device buffers (one inbound, one outbound) sized for
  `relay_max_active_slots * 4 * 4096 * sizeof(uint16_t)` FP16;
- a per-GPU descriptor table indexed by semantic tensor id;
- a per-GPU layer-id list (which transformer layers it owns);
- accessors only — kernel launches and decode wiring are deferred.

### Typed Resident Tensor Descriptor

```c
typedef enum {
    DS4_V100_SRC_BF16,
    DS4_V100_SRC_F32,
    DS4_V100_SRC_F8_E4M3_B128,
    DS4_V100_SRC_MXFP4,
    DS4_V100_SRC_I32
} ds4_v100_source_dtype;

typedef enum {
    DS4_V100_EXEC_DIAGNOSTIC,    /* BF16 gather/expand for inspection only */
    DS4_V100_EXEC_FP16_HMMA,     /* V100 tensor-core math; activations FP16 */
    DS4_V100_EXEC_FP32_CONTROL,  /* small reductions, norms, router scores */
    DS4_V100_EXEC_INT8_OR_LOWBIT,/* future low-bit kernel family */
    DS4_V100_EXEC_HOST_DEBUG,    /* never on hot path; for tests */
    DS4_V100_EXEC_UNSUPPORTED    /* explicit fail-closed marker */
} ds4_v100_exec_kind;

typedef enum {
    DS4_V100_KFAM_EMBEDDING_BF16,
    DS4_V100_KFAM_ATTN_CONTROL_F32,
    DS4_V100_KFAM_FP8_DEQUANT_F16_HMMA,
    DS4_V100_KFAM_MXFP4_GROUPED,
    DS4_V100_KFAM_SHARED_FP8_DENSE,
    DS4_V100_KFAM_ROUTER_F32_I32,
    DS4_V100_KFAM_HC_CONTROL_F32,
    DS4_V100_KFAM_OUTPUT_HEAD_BF16,
    DS4_V100_KFAM_UNKNOWN
} ds4_v100_kernel_family;

typedef struct ds4_v100_tensor_desc {
    const char            *semantic_id;
    ds4_v100_source_dtype  source_dtype;
    ds4_v100_kernel_family kernel_family;
    ds4_v100_exec_kind     exec_kind;       /* derived from policy */
    int                    owning_gpu;
    int                    layer_id;        /* -1 for global */
    uint64_t               arena_offset;
    uint64_t               byte_length;
    uint32_t               rows;
    uint32_t               cols;
    uint32_t               row_stride_elements;
    int64_t                scale_offset;    /* -1 if absent */
} ds4_v100_tensor_desc;

const ds4_v100_tensor_desc *
ds4_v100_context_find_desc(const ds4_v100_context *ctx,
                           const char *semantic_id);
int ds4_v100_context_for_each_desc(const ds4_v100_context *ctx,
                                   int (*cb)(const ds4_v100_tensor_desc *d, void *ud),
                                   void *ud);
```

Sprint 005 already defined `ds4_gpu_bf16_matrix_view`. The new descriptor
shape is a superset that adds source dtype, kernel family, exec kind, owning
GPU, and the policy-derived execution mode. The Sprint 005 view is kept
unchanged; Sprint 006 provides a converter
`ds4_v100_tensor_desc_to_bf16_view` so existing probes still work without
reaching into the context.

For Sprint 006 the descriptor table must cover at least:

- the global BF16 `token_embd.weight` (gpu0);
- one representative `f32_control` tensor (for example
  `blk.0.attn_kv_a_norm.weight`) on gpu0;
- one representative `f8_e4m3_b128` tensor (for example
  `blk.0.attn_kv_latent.weight`) on gpu0 — descriptor only, no kernel;
- one representative `mxfp4` tensor (for example
  `blk.0.ffn_down_exps.weight`) on gpu0 — descriptor only, no kernel;
- one representative HC control tensor family (the `hc_attn_*` triple)
  on gpu0.

Building descriptors for additional layers is allowed but not required for
SHIP. The classifier must reject any source dtype it does not yet recognize
rather than guessing — silent unknown-dtype passthrough is a Sprint 007 bug
waiting to happen.

### Execution-Format Policy Table

This sprint makes the V100 policy explicit and code-enforced. It is the
table the kernel sprints must register against. The classifier maps
`(source_dtype, kernel_family)` to `(exec_kind, notes)` and refuses any
pairing not in the table:

| Source Dtype | Kernel Family | Exec Kind | Notes |
|---|---|---|---|
| `bf16` | `ds4_embedding_bf16` | `DIAGNOSTIC` first; later expanded to FP16 at kernel boundary | Sprint 005 probe is the diagnostic path; production embedding expansion is Sprint 007 |
| `bf16` | `ds4_output_head_bf16` | `DIAGNOSTIC` first; later FP16 at kernel boundary | Output head is BF16 source; production GEMM must run FP16 HMMA |
| `f32` | `ds4_attention_control` | `FP32_CONTROL` | Small per-layer/per-head control tensors |
| `f32` | `ds4_router_f32_i32` | `FP32_CONTROL` | Router scores stay FP32 |
| `f32` | `ds4_hc_control_f32` | `FP32_CONTROL` | HC reduce/expand control tensors |
| `f8_e4m3_b128` | `v100_fp8_dequant_f16_hmma_pending` | `FP16_HMMA` | Unpack+HMMA kernel must be registered before exec is granted |
| `f8_e4m3_b128` | `v100_shared_fp8_dense_pending` | `FP16_HMMA` | Same |
| `mxfp4` | `v100_grouped_mxfp4_pending` | `INT8_OR_LOWBIT` | Grouped low-bit expert kernel; not implemented in this sprint |
| `i32` | `ds4_router_f32_i32` | `FP32_CONTROL` | Hash tables; treated as router auxiliary state |
| `bf16` | anything else | `UNSUPPORTED` | Fail closed — no implicit BF16 GEMM on V100 |
| `f8_e4m3_b128` | anything else | `UNSUPPORTED` | Fail closed |
| `mxfp4` | anything else | `UNSUPPORTED` | Fail closed |
| any | `UNKNOWN` | `UNSUPPORTED` | Fail closed |

The classifier API:

```c
int ds4_v100_classify(ds4_v100_source_dtype src,
                      ds4_v100_kernel_family kfam,
                      ds4_v100_exec_kind *out_exec,
                      const char **out_notes);
```

Sprint 007 (single-slot decode) will add a kernel registration table on top
of this classifier; this sprint only writes and tests the classifier.

### HC Relay Primitive

The DS4 layer map sends `[active_slots][4][4096]` FP16 across stage
boundaries, FP32 in debug. Sprint 006 implements the smallest credible
primitive:

```c
typedef struct {
    int      src_gpu;
    int      dst_gpu;
    uint32_t active_slots;
    int      use_fp32;   /* 0 = FP16 normal, 1 = FP32 debug */
} ds4_v100_relay_args;

int ds4_v100_relay_hc(const ds4_v100_context *ctx,
                      const ds4_v100_relay_args *args,
                      const void *src_dev_ptr,
                      void *dst_dev_ptr,
                      uint64_t bytes);
```

Semantics:

- If `cudaDeviceCanAccessPeer(dst_gpu, src_gpu)` and peer access has been
  enabled on the context, use `cudaMemcpyPeerAsync` on the source GPU's
  relay stream; synchronize on completion (or chain a recorded event for
  the next sprint).
- Otherwise, stage through a pinned-host bounce buffer in
  `ds4_v100_gpu_context` and report `pinned_host_fallback=true` in the
  context log.
- Reject `bytes` that exceed
  `active_slots * 4 * 4096 * (use_fp32 ? 4 : 2)`.
- Reject any `src_gpu == dst_gpu` call (HC relay never copies to itself).
- The context must always allocate enough relay buffer for the configured
  `relay_max_active_slots` so that callers do not allocate per-relay.

The primitive does not run decode logic. It is a transfer primitive whose
byte-equality the test harness can verify by uploading a deterministic FP16
pattern to the source buffer, relaying, and reading back the destination
buffer.

### Layer Skeleton Walker

```c
typedef struct {
    int      layer_id;
    int      stage;
    int      owning_gpu;
    uint32_t n_descs;
    uint32_t n_missing_descs;
    uint32_t n_unsupported_pairings;
    uint64_t weight_bytes;
} ds4_v100_layer_skeleton_row;

int ds4_v100_context_walk_layer_skeleton(const ds4_v100_context *ctx,
                                         ds4_v100_layer_skeleton_row *rows,
                                         uint32_t n_rows_capacity,
                                         uint32_t *n_rows_out);
```

The walker iterates layers 0-42 in stage order (per the 8-stage map in
`docs/architecture/DS4-V100-LAYOUT.md`), reports the owner, counts
descriptors actually built so far, and flags any tensor whose
`(source_dtype, kernel_family)` is classified `UNSUPPORTED`. It does **not**
execute attention or MoE math; it does not allocate KV; it does not invoke
RoPE; it does not call any of the existing `ds4_gpu_attention_*` or
`ds4_gpu_routed_moe_*` kernels.

For Sprint 006 the layer skeleton is satisfied if the walker visits all 43
layers and at least one layer reports a fully populated descriptor row set
for that layer's tensor families (with the rest reported as either
"descriptor not built this sprint" or "unsupported pairing"). The aim is to
exercise the iteration and validation surface, not to build descriptors for
every tensor.

### Relationship To Existing Code

- The new context **borrows** weight arenas via `ds4_gpu_arena_*`. It does
  not duplicate uploads.
- The new context **does not** call `ds4_gpu_set_model_map`,
  `ds4_gpu_cache_model_range`, or any `g_model_*` helper. Those remain the
  legacy single-runtime path.
- `ds4_engine_open` in `ds4.c` is **not** wired to construct the context by
  default. An optional `--v100-context-report <path>` inspect flag may be
  added to dump the context report alongside the existing pack reconcile
  output; this is the only `ds4.c` change considered in scope.
- The source-model generation guard in `ds4.c` is preserved verbatim.

### Memory And Reserve

Per-GPU footprint added by the context (above existing arena residency):

- 1 default stream + 1 relay stream + 1 cuBLAS handle: a few MiB of CUDA
  context overhead, mostly already counted by Sprint 004's reserve.
- Scratch: configurable per-GPU; default 256 MiB. Recorded in the report.
- Relay device buffers: `2 * active_slots * 4 * 4096 * 2` bytes for FP16.
  At `active_slots = 8` (planning default), that is about 1 MiB per
  direction per GPU, well under the reserve.
- Pinned-host bounce buffer: same size as the relay buffer; allocated only
  when peer access is unavailable.

The context reports these numbers per GPU. Any configuration that, when
added to the existing arena residency, drives a GPU within 256 MiB of the
32 GiB ceiling minus the declared reserve is a STOP.

---

## Implementation

### Phase 0: Build Hygiene And Sprint 005 Follow-Up

**Files:**

- `Makefile`
- `docs/sprints/SPRINT-006-REPORT.md` (create as scaffold when work starts)

**Tasks:**

- [ ] Confirm Sprint 005 local targets still build and pass:
      `make cpu`, `make tools/ds4-v100-pack`,
      `make tools/ds4-v100-residency-smoke`, `make tests/pack_index_smoke`,
      `make tests/gpu_arena_smoke`, `make tests/bf16_probe_smoke`.
- [ ] Confirm `./tests/pack_index_smoke`, `./tests/gpu_arena_smoke`,
      `./tests/bf16_probe_smoke`, `tests/residency_smoke_synthetic.sh` pass
      locally.
- [ ] Add `tools/ds4-v100-context-smoke`, `tests/v100_context_smoke`,
      `tests/cuda_v100_context_smoke`, and `tests/cuda_hc_relay_smoke`
      Makefile targets with appropriate `CUDA_ARCH=sm_70` guards.
- [ ] Record CUDA toolchain version, V100 pod recipe, and persistent pack
      directory path in the report scaffold.

**Kill gate:** none — this is housekeeping.

### Phase 1: Context Header, Descriptors, And Policy Classifier

**Files:**

- `ds4_v100_context.h` (new)
- `ds4_v100_context.c` (new; host/stub portions, no CUDA)
- `tests/v100_context_smoke.c` (new)
- `Makefile`

**Tasks:**

- [ ] Define `ds4_v100_context`, `ds4_v100_gpu_context`,
      `ds4_v100_tensor_desc`, `ds4_v100_context_options`, the source-dtype
      enum, the exec-kind enum, the kernel-family enum, and the
      `ds4_v100_init_mode` enum in `ds4_v100_context.h`.
- [ ] Implement `ds4_v100_classify` and back it with the policy table.
- [ ] Implement `ds4_v100_source_dtype_from_string` and
      `ds4_v100_kernel_family_from_string` helpers that parse the pack-index
      `source_dtype` and `kernel_family` columns.
- [ ] Implement `ds4_v100_context_open` for stub / no-CUDA builds that:
      reads the pack index via `ds4_pack_open`, builds descriptor tables for
      the required tensor families, runs the classifier on each, computes
      per-GPU descriptor counts and weight-byte totals, and returns the
      context.
- [ ] Implement `ds4_v100_context_find_desc`,
      `ds4_v100_context_for_each_desc`, and
      `ds4_v100_context_walk_layer_skeleton`.
- [ ] Implement `ds4_v100_context_print_report` that emits topology fields
      (visible vs PCI ids — stubbed to zero in non-CUDA builds), per-GPU
      arena-bytes / scratch-bytes / relay-bytes / reserve, descriptor counts
      per kernel family, and a policy summary row.
- [ ] Add `tests/v100_context_smoke.c` covering:
      classifier policy table (every supported pairing → expected exec
      kind); classifier rejects every documented unsupported pairing;
      descriptor build from a synthetic 5-tensor pack index; layer-skeleton
      walker visits expected layer rows and reports owning GPU correctly;
      invalid options (`expected_gpus == 0`, NULL pack path) fail closed;
      `mode == DS4_V100_INIT_PROBE_ONLY` does not require arenas.
- [ ] `git diff --check` passes; `make cpu` and the new smoke build clean.

**Kill gate:** STOP if the policy table cannot be defined without
contradicting `docs/architecture/DS4-V100-LAYOUT.md` for any tensor family
in the pack index — that signals a layout review, not a coding task.

### Phase 2: CUDA Context, Streams, Scratch, And Relay Buffers

**Files:**

- `ds4_v100_context_cuda.cu` (new)
- `ds4_v100_context.h` (extend with CUDA-side accessors via opaque ptrs)
- `tests/cuda_v100_context_smoke.c` (new)
- `Makefile`

**Tasks:**

- [ ] In `ds4_v100_context_cuda.cu`, implement the CUDA-backed
      `ds4_v100_context_open` overload that, after the host descriptor build,
      iterates GPUs `[0, expected_gpus)` and for each one:
      `cudaSetDevice(g)`, allocates the scratch buffer with `cudaMalloc`,
      allocates the two relay device buffers, creates a default stream and
      a relay stream, creates a cuBLAS handle and sets it to the default
      stream.
- [ ] Implement peer-access enablement: for each ordered pair `(i, j)` with
      `cudaDeviceCanAccessPeer(j, i)`, call `cudaSetDevice(i)` then
      `cudaDeviceEnablePeerAccess(j, 0)`. Record both the canAccess matrix
      and the actually-enabled matrix in the context.
- [ ] When peer access is unavailable for any required `(src, dst)` pair, the
      context allocates a pinned-host bounce buffer of `relay_bytes`
      capacity and marks the relay primitive's fallback path as active.
- [ ] Bracket every CUDA call with `cudaSetDevice(g)` and check return
      codes; any error during context open invalidates the partially-built
      context, frees what was allocated, and returns nonzero with a
      diagnostic.
- [ ] Implement `ds4_v100_context_print_report` for the CUDA build to also
      include `cudaMemGetInfo` (before/after context allocations), residency
      class for each piece (device vs pinned-host), and the P2P matrix.
- [ ] Add `tests/cuda_v100_context_smoke.c` that requires `>= 1` visible
      GPU, builds a probe-only context over `expected_gpus = min(visible, 8)`,
      verifies all per-GPU pieces are allocated as device memory (or pinned
      host for fallback), verifies the cuBLAS handle is set, verifies the
      scratch and relay buffers' addresses are device pointers via
      `cudaPointerGetAttributes`, and tears the context down cleanly.
- [ ] Run the synthetic CUDA test target locally (will be skipped if no GPU)
      and on the V100 pod under `CUDA_ARCH=sm_70`.

**Kill gate:** STOP if any allocation drives a GPU within 256 MiB of the
32 GiB ceiling after reserve. Adjust the architecture or shrink the scratch
default; do not just lower the reserve to make the number work.

### Phase 3: HC Relay Primitive

**Files:**

- `ds4_v100_context.h` (add `ds4_v100_relay_args` and
  `ds4_v100_relay_hc`)
- `ds4_v100_context.c` (host stub returns "no relay in stub")
- `ds4_v100_context_cuda.cu` (implement device-to-device and pinned-host
  fallback paths)
- `tests/cuda_hc_relay_smoke.c` (new)
- `Makefile`

**Tasks:**

- [ ] Implement `ds4_v100_relay_hc` with peer-async path when the pair has
      enabled peer access; otherwise stage through the pinned-host bounce
      buffer with `cudaMemcpyAsync` on each side.
- [ ] Reject `src_gpu == dst_gpu`, out-of-range GPU ids, oversized `bytes`,
      and NULL pointers with non-zero return.
- [ ] Add `tests/cuda_hc_relay_smoke.c`:
      requires `>= 2` visible GPUs; allocates a small destination context;
      writes a deterministic FP16 pattern (e.g. `value[i] = (uint16_t)i`)
      into the source device buffer; calls `ds4_v100_relay_hc` for
      `(src=0, dst=1)` with both FP16 and FP32-debug modes; reads back the
      destination via `cudaMemcpy` and asserts byte-equality with the
      pattern.
- [ ] Add a test variant that forces the pinned-host fallback (a CLI flag
      on the test binary, or a build define) so the fallback path is
      exercised on a single-GPU laptop with the CUDA build.
- [ ] When `>= 2` visible GPUs are not available, the test prints
      `cuda_hc_relay_smoke: skipped (n_gpus=N)` and exits 0 — never silently
      "passes" without doing work.

**Kill gate:** STOP if peer-access enablement fails on the V100 pod for a
pair that `cudaDeviceCanAccessPeer` reports as accessible — that's a real
bug. EXTEND if the pod has only one GPU available; record the missing
multi-GPU validation and ship the fallback test result.

### Phase 4: Standalone Context Smoke Tool

**Files:**

- `tools/ds4-v100-context-smoke.c` (new)
- `Makefile`
- `tests/v100_context_smoke.c` (extend if useful)

**Tasks:**

- [ ] Build a CLI:

      ```
      ds4-v100-context-smoke \
        --model /models/DSv4-Flash-256e-fixed.gguf \
        --index /workspace/ds4-pack/pack-index.tsv \
        --shard-dir /workspace/ds4-pack \
        --mode probe-only|use-existing-arenas|full-resident \
        --expected-gpus 8 \
        --scratch-mib 256 \
        --relay-active-slots 8 \
        --reserve-mib 3072 \
        --report /workspace/ds4/SPRINT-006-CONTEXT.log
      ```

- [ ] In probe-only mode, no shard upload happens; descriptors are built
      from the pack index alone.
- [ ] In use-existing-arenas mode (the V100 pod's normal case), arenas are
      opened against the existing residency pack but uploads are skipped
      (the persistent pack is assumed to be the same one Sprint 004
      uploaded; the smoke verifies arena sizes match `pack_payload_bytes`).
- [ ] In full-resident mode, the smoke runs the Sprint 004
      residency-smoke upload path before opening the context. This mode is
      optional and exists so the context smoke can be a standalone bring-up
      tool on a fresh pod.
- [ ] Output a deterministic context report including: visible vs PCI IDs;
      per-GPU arena size; per-GPU scratch size; per-GPU relay size;
      per-GPU `cudaMemGetInfo` deltas; P2P canAccess and enabled matrices;
      per-GPU descriptor counts by kernel family; layer-skeleton walker
      output (one row per layer); policy classifier summary (counts of each
      `exec_kind` across all built descriptors); guard status.
- [ ] On the V100 pod, run the tool with `--mode use-existing-arenas` and
      then with `--mode probe-only --expected-gpus 8`. Both must succeed
      and emit non-empty reports.

**Kill gate:** EXTEND if the persistent pack is missing from the pod and
cannot be regenerated within the sprint window.

### Phase 5: Cluster Validation And Archives

**Files:**

- `docs/sprints/drafts/SPRINT-006-CUDA-CONTEXT.log`
- `docs/sprints/drafts/SPRINT-006-CUDA-HC-RELAY.log`
- `docs/sprints/drafts/SPRINT-006-CONTEXT-PROBE.log`
- `docs/sprints/drafts/SPRINT-006-CONTEXT-RESIDENT.log`
- `docs/sprints/drafts/SPRINT-006-LAYER-SKELETON.log`
- `docs/sprints/drafts/SPRINT-006-POLICY.log`
- `docs/sprints/drafts/SPRINT-006-GUARD.log`

**Tasks:**

- [ ] On the V100 pod with `CUDA_ARCH=sm_70`, build:
      `make tests/v100_context_smoke tests/cuda_v100_context_smoke
      tests/cuda_hc_relay_smoke tools/ds4-v100-context-smoke`.
- [ ] Run `./tests/v100_context_smoke` and archive its output.
- [ ] Run `./tests/cuda_v100_context_smoke` and archive to
      `SPRINT-006-CUDA-CONTEXT.log`.
- [ ] Run `./tests/cuda_hc_relay_smoke` and archive to
      `SPRINT-006-CUDA-HC-RELAY.log`. Confirm device-to-device byte
      equality. If the pod has only one GPU, run the fallback variant and
      mark the multi-GPU case as EXTEND.
- [ ] Run `./tools/ds4-v100-context-smoke --mode probe-only ...` and archive
      to `SPRINT-006-CONTEXT-PROBE.log`.
- [ ] Run `./tools/ds4-v100-context-smoke --mode use-existing-arenas ...`
      and archive to `SPRINT-006-CONTEXT-RESIDENT.log`.
- [ ] Extract the layer-skeleton walker section into
      `SPRINT-006-LAYER-SKELETON.log` and the policy summary into
      `SPRINT-006-POLICY.log`.
- [ ] Confirm `./ds4 -m /models/DSv4-Flash-256e-fixed.gguf "hi"` still fails
      closed with the expected source-model guard message and archive to
      `SPRINT-006-GUARD.log`.
- [ ] Delete the disposable pod after copying artifacts back into the repo.

**Kill gate:** STOP if any GPU exceeds budget after context allocations are
added on top of arena residency; STOP if peer access fails where
`canAccessPeer` succeeded; STOP if the layer skeleton reports any
classifier-flagged unsupported pairing for the in-scope descriptor set
(which would indicate a policy-table bug).

### Phase 6: Sprint Report And Follow-Ups

**Files:**

- `docs/sprints/SPRINT-006-REPORT.md`
- `docs/sprints/SPRINT-006-FOLLOWUPS.md` (if needed)
- `docs/sprints/SPRINT-006-DEFERRED.md`
- `docs/sprints/VISION.md`

**Tasks:**

- [ ] Record verdict (SHIP / EXTEND / STOP) with evidence pointers.
- [ ] Enumerate the API surface Sprint 007 can target:
      tensor descriptor lookups, scratch acquisition pattern, relay
      semantics, the kernel-registration table needed for FP16-HMMA over
      FP8 unpack.
- [ ] List Sprint 007 prerequisites that this sprint did not satisfy
      (e.g., no FP8 unpack kernel registered; no MXFP4 grouped kernel
      registered; no KV allocation; no slot scheduler).
- [ ] Update `docs/sprints/VISION.md` Sprint 006 entry with outcome.
- [ ] Preserve source-model generation guard status in the report.

**Kill gate:** none — always runs.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_v100_context.h` | Create | Opaque context, descriptor structs, policy enums, classifier API, relay API, layer-skeleton API |
| `ds4_v100_context.c` | Create | Host/stub implementation: pack-index → descriptor table, classifier, layer-skeleton walker, context report |
| `ds4_v100_context_cuda.cu` | Create | CUDA-backed context: streams, cuBLAS handles, scratch, relay buffers, peer access, pinned-host fallback |
| `ds4_gpu.h` | Read-only | Existing arena API is borrowed; no changes in this sprint |
| `ds4_cuda.cu` | Read-only | Legacy single-runtime globals untouched |
| `ds4_pack.h` / `ds4_pack.c` | Read-only | Existing pack-index API is the descriptor source |
| `ds4.c` | Read-only (optional inspect flag if cheap) | Source-model generation guard preserved |
| `tools/ds4-v100-context-smoke.c` | Create | Standalone diagnostic CLI |
| `tests/v100_context_smoke.c` | Create | Model-less context + policy + layer-skeleton tests |
| `tests/cuda_v100_context_smoke.c` | Create | Direct-CUDA context allocation tests |
| `tests/cuda_hc_relay_smoke.c` | Create | Direct-CUDA HC relay primitive test |
| `Makefile` | Modify | New tool and test targets; CUDA_ARCH=sm_70 guards |
| `docs/sprints/SPRINT-006-REPORT.md` | Create | Execution report |
| `docs/sprints/SPRINT-006-FOLLOWUPS.md` | Create if needed | Follow-up surface for Sprint 007 |
| `docs/sprints/SPRINT-006-DEFERRED.md` | Create | Items discussed but excluded from scope |
| `docs/sprints/VISION.md` | Modify | Record Sprint 006 outcome and refine Sprint 007 framing |
| `docs/sprints/drafts/SPRINT-006-CUDA-CONTEXT.log` | Create on cluster | Direct-CUDA context test artifact |
| `docs/sprints/drafts/SPRINT-006-CUDA-HC-RELAY.log` | Create on cluster | HC relay byte-equality artifact |
| `docs/sprints/drafts/SPRINT-006-CONTEXT-PROBE.log` | Create on cluster | Probe-only context report |
| `docs/sprints/drafts/SPRINT-006-CONTEXT-RESIDENT.log` | Create on cluster | Use-existing-arenas context report |
| `docs/sprints/drafts/SPRINT-006-LAYER-SKELETON.log` | Create on cluster | Layer-skeleton walker output |
| `docs/sprints/drafts/SPRINT-006-POLICY.log` | Create on cluster | Policy classifier summary |
| `docs/sprints/drafts/SPRINT-006-GUARD.log` | Create on cluster | Source-model generation guard artifact |
| `docs/architecture/DS4-V100-LAYOUT.md` | Read; modify only on STOP | Layout/policy reference |

---

## Definition Of Done

- [ ] `ds4_v100_context.h` defines the opaque context, per-GPU context,
      tensor descriptor, source-dtype enum, kernel-family enum, exec-kind
      enum, init-mode enum, options struct, relay-args struct, classifier
      API, descriptor accessor API, layer-skeleton walker, and relay
      primitive declarations.
- [ ] `ds4_v100_context.c` implements descriptor build from the pack index,
      the classifier, the layer-skeleton walker, the context report, and
      the host/stub init/teardown.
- [ ] `ds4_v100_context_cuda.cu` allocates per-GPU streams, cuBLAS handles,
      scratch, and relay buffers; enables peer access where supported;
      provisions a pinned-host bounce buffer where it is not.
- [ ] The classifier maps every supported `(source_dtype, kernel_family)`
      pair to the documented exec kind and rejects every unsupported pair
      with a non-zero return.
- [ ] Descriptors are built for at least: `token_embd.weight` (BF16, gpu0),
      one F32 control tensor, one F8_E4M3_B128 tensor, one MXFP4 tensor,
      one HC F32 control triple — all on a real layer.
- [ ] `ds4_v100_context_walk_layer_skeleton` visits all 43 layers in stage
      order and reports owning GPU correctly per
      `docs/architecture/DS4-V100-LAYOUT.md`.
- [ ] The HC relay primitive transfers a deterministic FP16 payload
      byte-equal between two devices (or via pinned-host fallback) and
      respects `[active_slots][4][4096]` bounds.
- [ ] V100 execution-format policy is encoded in code, summarized in the
      context report, and emitted as a separate `SPRINT-006-POLICY.log` on
      the cluster.
- [ ] No kernel registers BF16, FP8, or FP4 as a native V100 tensor-core
      execution format; the classifier's `UNSUPPORTED` outcome is wired
      and tested.
- [ ] Local tests (`tests/v100_context_smoke`, plus Sprint 005 tests
      still green) pass on the laptop.
- [ ] CUDA synthetic tests (`tests/cuda_v100_context_smoke`,
      `tests/cuda_hc_relay_smoke`) build with `CUDA_ARCH=sm_70` and pass
      on the V100 pod (multi-GPU relay may EXTEND on single-GPU pods, with
      the missing validation explicitly recorded).
- [ ] Real-pack context smoke (`tools/ds4-v100-context-smoke
      --mode use-existing-arenas`) runs on the V100 pod and emits the
      documented report fields.
- [ ] No persistent dequantized FP16/F32 copy of any large source tensor is
      created in this sprint.
- [ ] The source-model generation guard remains active; `./ds4 -m ...
      "hi"` still fails closed.
- [ ] `git diff --check` passes.
- [ ] `docs/sprints/SPRINT-006-REPORT.md` records verdict, evidence
      pointers, the API surface Sprint 007 can rely on, and the
      explicitly-deferred next-sprint surface.
- [ ] `docs/sprints/VISION.md` Sprint 006 entry is updated with outcome.

---

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| The new context drifts into a decode rewrite (HC math, kernel registration, KV) | High | High | Header explicitly defers kernel registration to Sprint 007; the layer-skeleton walker has a no-math contract; smoke tool rejects any kernel-launch flag |
| Pinned-host fallback masks a real peer-access bug | Medium | High | Always report `peer_access_enabled` per pair; the cluster log must show device-to-device on the V100 pod where `cudaDeviceCanAccessPeer` returns 1; only fall back when canAccess is 0 |
| Policy table contradicts the layout document silently | Medium | High | Phase 1 unit test pins the policy table to the layout document by spelling out every pairing; STOP if a tensor family in the pack index has no entry |
| BF16/FP8/FP4 sneaks back in as a "native V100 compute format" via a permissive default | Medium | High | Classifier defaults to `UNSUPPORTED`; every supported pairing is enumerated; tests assert that BF16 outside `DIAGNOSTIC`/`FP16_HMMA` returns `UNSUPPORTED` |
| Scratch + relay allocation drives a GPU below reserve when added to arena residency | Medium | Medium | Phase 5 report records pre/post `cudaMemGetInfo`; STOP if any GPU is within 256 MiB of ceiling minus reserve; reduce default scratch before retrying |
| Multi-GPU node unavailable; only a single-GPU pod is reachable | Medium | Medium | EXTEND outcome with pinned-host fallback path validated and the multi-GPU relay case explicitly missing |
| Context coexistence with `g_cublas`, `g_cuda_tmp`, and `g_model_*` causes hidden state collisions | Medium | High | New context never calls legacy helpers; every CUDA op in the context brackets `cudaSetDevice`; tests open/close the context with the legacy path absent and with it present |
| Layer-skeleton walker becomes a scheduler | Medium | Medium | Walker returns rows of metadata; it does not own streams beyond reading the context; no `attention`, `moe`, or `kv` symbols appear in `ds4_v100_context*` |
| Descriptor schema locks in a bad shape for Sprint 007 | Medium | Medium | Descriptor is additive over Sprint 005's matrix view; Sprint 007 can extend without rewriting; smoke tool exposes the descriptor via report only |
| cuBLAS handle creation fails on V100 pod under CUDA 12.x | Low | Medium | Capture the exact CUDA + cuBLAS versions in the report; document the failure with the cuBLAS error string; retry with the standard `homelab-k8s-dev` image |
| Persistent pack directory on the pod is stale | Low | Medium | Compare arena bytes against `pack_payload_bytes(gpu)`; the smoke fails closed on any mismatch and the operator regenerates shards (Sprint 004 path) |

---

## Security Considerations

- The context treats pack-index inputs, source GGUF metadata, and CLI
  options as untrusted local inputs. Every numeric field (offsets, byte
  lengths, GPU ids, layer ids) is bounds-checked before use.
- All `cudaMalloc`, `cudaMemcpyPeerAsync`, and `cudaMemcpyAsync` calls
  validate byte counts against the relay/scratch capacity and the source
  arena range before issue.
- Relay does not expose device pointers across the public API. Callers pass
  device pointers they already own; the context never returns raw arena
  pointers.
- Pinned-host bounce buffers are allocated read/write only inside the
  context's lifetime and are freed on close, including on partial-failure
  cleanup.
- The context report writes to a caller-provided file path; the smoke
  rejects path traversal and refuses to overwrite the source model, the
  pack index, or any shard file.
- The HTTP server in `ds4_server.c` is not modified. The new context is
  not exposed across any network surface.
- The source-model generation guard in `ds4.c` is preserved verbatim. The
  context's existence does not unlock decode.
- Integer overflow in
  `active_slots * 4 * 4096 * sizeof_format` and
  `scratch_bytes_per_gpu + relay_bytes` is checked with overflow-safe
  arithmetic and rejected on overflow.

---

## Dependencies

- Sprint 004: `ds4_pack` API, `ds4_gpu_arena_*` API, residency proof on the
  V100 pod, persistent pack at `/workspace/ds4-pack` (or equivalent).
- Sprint 005: BF16 row-gather/expand probe, the BF16 matrix view contract
  the descriptor superset extends, and the policy correction that V100 has
  no native BF16/FP8/FP4 tensor-core execution.
- Source model at `/models/DSv4-Flash-256e-fixed.gguf` (cluster).
- 8x V100-SXM2-32GB pod with CUDA 12.x and `sm_70` build environment.
- Cluster operating procedure
  `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`.
- `docs/architecture/DS4-V100-LAYOUT.md` for the 8-stage layer map and the
  `[active_slots][4][4096]` HC payload shape.

---

## Open Questions

1. **Full real per-GPU weight arenas or metadata/probe-only?** This draft
   proposes a three-way `ds4_v100_init_mode`:
   `probe-only` (descriptors only, no arenas), `use-existing-arenas`
   (descriptors plus references to Sprint 004's residency pack), and
   `full-resident` (Sprint 004 upload before context init). The cluster
   verdict for SHIP should be `use-existing-arenas` because the V100 pod
   already has a persistent residency pack and a fresh full upload would
   pay a long upload cost for no new evidence. Probe-only is the model-less
   default for laptop tests.

2. **CUDA peer access required, or pinned-host fallback in the same sprint?**
   This draft includes **both**. Peer access is the production path and
   must be exercised on the V100 pod. The pinned-host fallback exists so
   the primitive has a documented behavior on hardware that does not enable
   peer between a pair, and so single-GPU laptops with the CUDA build can
   still exercise the staging path. The relay test asserts byte equality
   on whichever path is active and the context report records which path
   was used.

3. **How much of `ds4.c` should know about the new context?**
   Engine-open wiring is **out** of this sprint's default scope. An optional
   `--v100-context-report <path>` inspect flag may be added if it is cheap
   and does not pull in decode wiring. Production wiring waits for
   Sprint 007's first kernel.

4. **Where do typed descriptors live?**
   This draft places them in a new `ds4_v100_context.h` rather than
   overloading `ds4_gpu.h` (which is general GPU/Metal-shared API) or
   `ds4_pack.h` (which is metadata, not runtime). This keeps the
   residency-only Sprint 004 API and the appliance-runtime Sprint 006
   API cleanly separable.

5. **Minimal layer skeleton output: report only, or no-math walk?**
   This draft chooses **walker + per-layer row report**. The walker visits
   all 43 layers in stage order, reports the owning GPU and a per-row
   summary, and validates ownership against
   `docs/architecture/DS4-V100-LAYOUT.md`. At least one layer must have a
   fully-populated descriptor row set; other layers may report
   "descriptor not built this sprint" without failing the run. No
   attention, MoE, or KV math is executed.

Remaining merge-time questions:

- Default `scratch_bytes_per_gpu` (256 MiB proposed) versus a smaller
  value (64 MiB) to leave more reserve. Cluster `cudaMemGetInfo` data
  should pick the number.
- Default `relay_max_active_slots` (8 proposed). Sprint 007 will revisit
  once decode batching is understood.
- Should the context report emit JSON in addition to TSV? Defer to
  Sprint 007 unless a downstream consumer asks for JSON.
- Should the policy table live in code only, or also in a checked-in
  `docs/architecture/DS4-V100-POLICY.md`? This draft proposes code-only
  with the layout document remaining the human-readable anchor; if drift
  becomes a recurring risk, a generated policy table can be added later.
