# SPRINT-008 - Source Oracle Harness And V100 KV Anchors (CLAUDE draft)

**Status:** DRAFT 2026-05-18
**Predecessor:** SPRINT-007 (guarded source-layout single-slot decode oracle; `--source-layout-oracle` plus `--dump-logprobs` selected the expected official first token `16` for `short_reasoning_plain`).
**Successor:** Prompt prefill, multi-token oracle decode, and the first production-relevant V100 source-format kernel.

---

## Overview

Sprint 007 shipped a guarded CPU-only source-layout oracle. Normal generation
on `/models/DSv4-Flash-256e-fixed.gguf` still fails closed; an opt-in
`--source-layout-oracle --dump-logprobs ...` diagnostic session can run a
single first-token decode and was hand-verified against the
`short_reasoning_plain` official vector. The MXFP4 row layout was corrected
to match GGML `block_mxfp4` nibble ordering, and source-layout KV was reset
to the F16 baseline.

Sprint 008 turns that one-off proof into a **repeatable harness** and lays
the first **F16 KV admission surface** plus a **conservative device-side
anchor** that production prefill/decode work can build on without weakening
the existing guard.

The sprint has four bounded outcomes:

1. An automated official-vector runner that drives the existing
   `--source-layout-oracle --dump-logprobs` path on at least one short
   fixture and asserts selected-token equality from a test target, removing
   the manual JSON inspection step.
2. Targeted guard tests that lock in the Sprint 007 invariants: normal
   source-layout open fails closed, oracle-with-MTP fails closed,
   oracle-on-non-CPU fails closed, and oracle-without-diagnostic-session
   refuses session creation.
3. A concrete F16 KV budget/admission surface on `ds4_v100_context` that
   classifies each layer as SWA-only, ratio-4 (+indexer), or ratio-128 and
   computes per-slot/per-stage planned KV bytes against the existing
   per-GPU reserve. KV stays a planning/admission concept; this sprint
   does not populate device KV bytes.
4. One device-side anchor that exercises CUDA on the V100 pod without
   claiming production throughput or unlocking source generation. The
   conservative choice is a CUDA BF16 source-row-to-F32 probe over a
   resident `token_embd.weight` slice, verified bit-exact against the
   existing host helper `ds4_src_bf16_row_to_f32`.

V100 precision policy is unchanged: BF16/FP8/MXFP4 are source/packed
inputs only, FP16 HMMA with FP32 accumulation is the production target,
FP32 is reserved for control and oracle math, and F16 KV is the only
baseline cache. The sprint does not introduce a production FP8/MXFP4 dense
or grouped kernel and does not unlock normal generation.

The hardest single risk is scope drift: prefill, indexer growth, real
device KV population, or kernel scheduling all look "next" but each one
defeats the small, verifiable contract this sprint relies on. Every phase
ends in an explicit kill gate.

---

## Outcome Contract

- **SHIP** if: the official-vector runner runs from `make test`-style
  invocation on the cluster and asserts selected-token equality for at
  least `short_reasoning_plain` against `official/short_reasoning_plain.official.json`;
  guard regression tests cover normal-rejection, MTP-rejection,
  non-CPU-rejection, and session-without-diagnostic-unlock rejection;
  `ds4_v100_context` exposes per-layer-class F16 KV bytes and per-stage
  planned KV totals that fail closed when the reserve cannot be honored;
  `v100_context_smoke` covers the KV admission surface; one CUDA device-side
  anchor (BF16 source row equality on resident bytes) builds and passes on
  the 8x V100 pod under `CUDA_ARCH=sm_70`; normal source-layout open still
  fails closed; logs are archived under `docs/sprints/drafts/SPRINT-008-*`;
  `SPRINT-008-REPORT.md` is written.

- **EXTEND** if: the runner and guard tests land and the KV admission
  surface lands locally, but the cluster device-side anchor is blocked by
  pod availability, CUDA build environment, or unexplained timing
  variance. In that case the sprint ships items 1, 2, and 3 with cluster
  evidence for the runner and host tests for the KV surface, and records
  the device-anchor blocker in the report.

- **STOP** if: the existing diagnostic unlock cannot be relied on (e.g. the
  `--dump-logprobs` session gate turns out to weaken the source-layout
  guard under one of the new tests), the F16 KV admission numbers do not
  fit the documented per-GPU reserve under the layer-owned topology, or
  the implementation needs to grow into prefill, multi-slot, indexer
  population, MTP, or a production FP8/MXFP4 kernel to make any single
  item pass.

---

## Use Cases

Each phase produces value even if a later phase slips:

| Phase | Useful output if sprint stops here |
|---|---|
| P0 | Sprint 007 surfaces re-validated locally and on the cluster; the existing `g_source_layout_oracle_f16_kv` global is documented and exercised by a model-less test so its behavior is not silently lost. |
| P1 | MXFP4 parity test asserts the low-half/high-half nibble layout against the in-tree dequant reference so the Sprint 007 correction cannot regress. |
| P2 | Source-layout guard regression tests fail closed for normal generation, MTP sidecar, non-CPU backend, and session-without-diagnostic-unlock paths. |
| P3 | `ds4_v100_context` reports per-layer-class F16 KV bytes (SWA-only, ratio-4 + indexer, ratio-128) and per-stage planned-KV totals; production-topology open fails closed when KV totals push the per-GPU reserve below the configured floor. |
| P4 | Automated official-vector runner asserts selected-token equality for `short_reasoning_plain` against `official/short_reasoning_plain.official.json` and emits a JSON evidence artifact. |
| P5 | CUDA BF16 source-row probe runs on the 8x V100 pod and asserts bit-exact equality against the host `ds4_src_bf16_row_to_f32` helper on resident `token_embd.weight` bytes. |
| P6 | `SPRINT-008-REPORT.md` archives verdict, cluster artifacts, the API surface Sprint 009 can rely on, and a precise statement of what was *not* proven (prefill, decode throughput, indexer population, deployment). |

---

## Architecture

### Source Of Truth

`docs/architecture/DS4-V100-LAYOUT.md` remains the architecture anchor.
The KV admission surface consumes the layer-class table verbatim:

- layers 0-1 are SWA-only (0.125 MiB raw SWA per slot per layer, no
  compressed attention or indexer);
- ratio-4 layers (even 2..42) carry compressed attention + indexer KV
  (~256.1 MiB `attn_kv` + ~64.0 MiB `indexer_kv` per slot per layer at 1M);
- ratio-128 layers (odd 3..41) carry compressed attention only (~8.1 MiB
  per slot per layer at 1M).

Sprint 007 left source-layout KV at F16. This sprint codifies that as the
only baseline KV format. F8 KV remains an explicit non-goal until prefill
correctness exists.

`tests/test-vectors/official/short_reasoning_plain.official.json` and
`tests/test-vectors/prompts/short_reasoning_plain.txt` are the SHIP fixture.
The runner may opportunistically attempt a second short fixture (e.g.
`short_italian_fact`) on the cluster if the first passes within time.

### Module Boundary

```text
ds4_v100_context.h / .c
    + ds4_v100_layer_class:
        DS4_V100_LAYER_CLASS_SWA_ONLY,
        DS4_V100_LAYER_CLASS_RATIO_4,
        DS4_V100_LAYER_CLASS_RATIO_128;
    + ds4_v100_layer_info.layer_class;
    + ds4_v100_context_options.kv_ctx_tokens (per-slot context budget);
    + ds4_v100_context_options.kv_active_slots;
    + ds4_v100_stage_info.planned_kv_bytes is populated from the layer
      classes and admission inputs instead of accepting an opaque value;
    + ds4_v100_kv_class_bytes_per_slot(class, ctx) helper;
    + ds4_v100_context_planned_kv_bytes(ctx, stage_id) accessor;
    + KV admission step in validate_memory_budget that fails closed when
      reserve_bytes cannot be honored after KV totals;
    + KV_CACHE family classifier flips from EXEC_UNSUPPORTED to
      EXEC_DIAGNOSTIC_ONLY when source_layout_oracle is requested AND the
      F16 KV baseline is the only mode planned. (No production-runtime
      label is granted here.)

ds4_v100_context_cuda.cu  (read-mostly)
    + ds4_v100_cuda_bf16_source_row_probe(...): pushes a resident BF16
      row from `token_embd.weight` through a small CUDA kernel that
      converts bf16 -> f32 using the same bit pattern as
      ds4_src_bf16_to_f32, copies to host, and returns the device F32
      slice for bit-equality comparison against the host helper. No
      persistent dequantized copy; the device F32 lives in bounded
      scratch.

tests/source_oracle_runner.c
    Drives `--source-layout-oracle --dump-logprobs ...` over a short
    fixture, parses the emitted JSON, and asserts selected-token id
    equality against `tests/test-vectors/official/<case>.official.json`.
    Cluster-only by default; gated on the model path being present.

tests/source_layout_guard_smoke.c
    Model-less guard regression: a synthetic minimal `ds4_engine_options`
    setup that exercises:
      - normal open of a source-layout-flagged fixture fails closed with
        the Sprint 007 guard message;
      - oracle + MTP fails closed;
      - oracle + non-CPU backend fails closed;
      - oracle without --dump-logprobs (no diagnostic-session unlock)
        refuses ds4_session_create with the existing message.
    Where exercising the full engine open is too expensive without a
    real model, the test stubs the minimal config needed to trigger
    `model_uses_v100_source_layout` so the guard paths execute.

tests/mxfp4_parity_smoke.c
    Direct byte-equality between ds4_src_mxfp4_row_to_f32 and an inline
    minimal block_mxfp4 reference (low-half/high-half) over random and
    edge-case bytes. Locks the Sprint 007 layout correction.

tests/v100_context_smoke.c (extend)
    Add KV admission cases: SWA-only/ratio-4/ratio-128 class totals at
    representative ctx sizes; production-topology open fails closed when
    planned KV + scratch + relay + reserves push past device totals.

tests/cuda_v100_kv_anchor_smoke.c
    Cluster CUDA test that allocates a small BF16 row on device,
    runs the device probe, and asserts bit-exact equality with
    ds4_src_bf16_row_to_f32 over the same bytes.

Makefile
    New CPU test targets:
      tests/source_layout_guard_smoke
      tests/mxfp4_parity_smoke
      tests/source_oracle_runner
    New CUDA test target:
      tests/cuda_v100_kv_anchor_smoke
    `make test-source-oracle-runner` wraps the runner invocation when a
    model and the cluster path are present.
```

### Layer-Class And KV Admission

`ds4_v100_stage_for_layer` already encodes the 8-stage map. This sprint
adds the third axis: layer class. The class is fixed by layer id under
the architecture anchor:

| layer id | class |
|---:|---|
| 0, 1 | SWA-only |
| even 2..42 | ratio-4 + indexer |
| odd 3..41 | ratio-128 |

`ds4_v100_kv_class_bytes_per_slot(class, ctx_tokens)` computes the F16 KV
bytes contributed by one layer of that class at the given context using
the layout document's numbers:

- SWA-only: `128 * DS4_N_HEAD_DIM * sizeof(f16)` raw SWA per slot per
  layer (architecture: ~0.125 MiB per slot per layer).
- ratio-4: raw SWA + `comp_attn_bytes(ctx)` + `indexer_kv_bytes(ctx)`
  (architecture: at 1M, ~256.1 MiB `attn_kv` + ~64.0 MiB `indexer_kv` per
  slot per layer).
- ratio-128: raw SWA + `comp_attn_bytes(ctx)` (architecture: at 1M,
  ~8.1 MiB per slot per layer).

`comp_attn_bytes`, `indexer_kv_bytes`, and `raw_swa_bytes` are derived
from `DS4_N_HEAD_DIM`, `DS4_N_INDEXER_HEAD_DIM` and existing constants
in `ds4.c`. They are pure planning formulas; they do not allocate device
memory and do not write into `ds4_kv_cache`. The legacy CPU
`kv_cache_init` is not modified.

`ds4_v100_context_options` gains:

```c
uint64_t kv_ctx_tokens;     /* per-slot context capacity for planning */
uint64_t kv_active_slots;   /* number of slots whose KV is admitted */
```

`validate_memory_budget` aggregates per-layer KV per stage using the new
classifier, multiplies by `kv_active_slots`, and uses that as
`planned_kv_bytes` (overriding the previous opaque caller value if
caller passed both — caller wins only if `kv_ctx_tokens == 0`, which is
the back-compat default that exercises Sprint 006 behavior). It then
re-runs the existing reserve check.

The KV admission surface is fail-closed: if any stage cannot honor
`reserve_bytes_per_gpu` after KV totals are added, the open returns
nonzero with a precise message.

### Source-Layout Guard And Diagnostic Unlock

This sprint does **not** add a second unlock token. Sprint 007 shipped a
two-flag diagnostic unlock (`--source-layout-oracle` plus
`--dump-logprobs`, which sets `source_layout_oracle_sessions = true`).
The cost/benefit of adding an opaque code-level token is low and was
flagged as Open Question 2 in the intent; the answer in this draft is
"keep the CLI gate, lock it in with tests" rather than "add a new code
constant".

Tests assert:

- `ds4_engine_open` with the source-layout fixture and no oracle option
  returns 1 with the Sprint 007 guard text.
- `ds4_engine_open` with oracle + MTP returns 1 with the MTP rejection
  text.
- `ds4_engine_open` with oracle + non-CPU backend returns 1 with the
  backend rejection text.
- `ds4_session_create` on an oracle-only engine without
  `source_layout_oracle_sessions` returns 1 with the existing session
  rejection text.

The guard test does not require the 145 GiB source model. It constructs
the minimal model state that makes `model_uses_v100_source_layout` and
`weights_validate_v100_source_layout` return true, calls `ds4_engine_open`
with each option combination, and asserts the return code and stderr
text. If a fully minimal in-memory fixture is too invasive to set up,
the test wraps `ds4` as a subprocess against a small synthetic GGUF
fixture committed under `tests/test-vectors/` (preferred path: in-memory).

### Automated Official-Vector Runner

`tests/source_oracle_runner.c` is a thin driver that:

1. fork/execs the local `./ds4` binary with the exact Sprint 007
   incantation:
   ```sh
   ./ds4 --cpu --source-layout-oracle -t 80 \
     -m $DS4_TEST_MODEL --nothink \
     --prompt-file tests/test-vectors/prompts/<case>.txt \
     -n 1 -c 4096 \
     --dump-logprobs /tmp/ds4-source-oracle-<case>-<pid>.json \
     --logprobs-top-k 20
   ```
2. parses the emitted JSON (no jq dependency; small tolerant reader);
3. extracts `steps[0].selected.id` and `steps[0].selected.text`;
4. parses `tests/test-vectors/official/<case>.official.json` for the
   expected first token id/text;
5. asserts equality and writes a fixed-name evidence file under
   `docs/sprints/drafts/SPRINT-008-cluster-logs/SPRINT-008-oracle-<case>.json`
   when run on the cluster.

The runner is **gated on environment**:

- `DS4_TEST_MODEL` and `DS4_TEST_VECTOR_FILE`-style env vars decide
  whether the runner attempts a real run or skips with a clear "skipped:
  model not present" message that does not fail the test.
- The Makefile target `make test-source-oracle-runner` does the
  build + invocation. `make test` does not include it by default to
  keep laptop iteration fast.

The default fixture is `short_reasoning_plain` because Sprint 007 already
verified it. A second fixture is opportunistic and reported as a
non-blocking signal.

### Conservative Device-Side Anchor

The intent's Open Question 1 asks whether the first device-side anchor
should be the KV admission/guard surface or a CUDA kernel probe. This
draft chooses **both, narrowly**: the KV admission surface ships as a
host-side planning surface (no device allocation), and the device-side
anchor is a CUDA BF16 source-row probe that touches resident bytes and
validates the host source-format helper on the GPU. The choice avoids
production kernel work while still proving that source bytes can be
exercised on the V100 in this sprint.

`ds4_v100_cuda_bf16_source_row_probe`:

- accepts a resident `token_embd.weight` BF16 slice (offset + nrows +
  ncols) from the Sprint 005 BF16 probe surface;
- launches a tiny CUDA kernel that performs `__bf162float` (or the
  bit-shifted equivalent `(uint32_t)bf16 << 16` reinterpreted as float)
  per element into device scratch;
- `cudaMemcpy`s the device F32 scratch back to the host;
- the test asserts each F32 word is bit-equal to
  `ds4_src_bf16_to_f32(bf16_word)` on the host;
- scratch is freed at end of test. No persistent device F32 copy is
  retained.

This anchor is correctness-only. It does not measure throughput, does
not exercise a production HMMA path, and does not unlock source
generation. It is the smallest device-side proof that source bytes can
be exercised end-to-end on the V100 in Sprint 008 without committing to
kernel scheduling work that belongs in a later sprint.

If a real `token_embd.weight` slice on the V100 pod is not available in
the time budget, the anchor falls back to a synthetic BF16 buffer
allocated and uploaded inline. The cluster log records which mode ran.

---

## Implementation

### Phase 0: Re-Baseline And Local Health

**Files:**

- `Makefile`
- (read-only) `ds4.c`, `ds4_v100_context.c`, `ds4_source_formats.c`,
  `tests/source_dtypes_smoke.c`, `tests/v100_context_smoke.c`,
  `docs/sprints/SPRINT-007-REPORT.md`,
  `docs/sprints/drafts/SPRINT-007-cluster-logs/`

**Tasks:**

- [ ] Confirm Sprint 007 local targets still build and pass:
      `make cpu`, `make tests/source_dtypes_smoke`,
      `make tests/v100_context_smoke`,
      `./tests/source_dtypes_smoke`,
      `./tests/v100_context_smoke`,
      `git diff --check`.
- [ ] Confirm the existing source-layout-oracle CLI flag still exists
      at `ds4_cli.c:1321-1322` and `ds4.c:17717-17738`; record the
      exact stderr texts in scratch notes so guard tests assert against
      stable strings.
- [ ] Re-read the Sprint 007 cluster artifact
      `SPRINT-007-source-oracle-official-short-reasoning.json` and lock
      in the expected SHIP comparison numbers (id=926, text="16",
      bytes=[49,54]).
- [ ] Record the persistent cluster model path (currently
      `/models/DSv4-Flash-256e-fixed.gguf`) and pack scratch path used by
      Sprint 007 so the runner can use the same.

**Kill gate:** STOP if any Sprint 007 local target fails or if the
diagnostic unlock CLI surface has changed in an unexpected way that
would force a re-design rather than a small adjustment.

### Phase 1: MXFP4 Parity Hardening

**Files:**

- `tests/mxfp4_parity_smoke.c` (create)
- `Makefile`

**Tasks:**

- [ ] Add `tests/mxfp4_parity_smoke.c`. The test constructs a small
      synthetic 64-element MXFP4 row (two blocks) with hand-picked
      scale bytes and nibble bytes that exercise:
      low nibble in first half / high nibble in second half,
      negative zero/positive zero codes, max positive (0x7),
      max negative (0xf), and at least one scaled block.
- [ ] Define an inline `block_mxfp4_reference_row(...)` in the test that
      hard-codes the GGML low-half/high-half nibble ordering. Walk
      `ds4_src_mxfp4_row_to_f32` against the inline reference and assert
      byte-equality of the resulting F32 outputs.
- [ ] Add a `dot` parity check using `ds4_src_mxfp4_row_dot` against an
      independent reference dot computed from the row-to-F32 helper.
- [ ] Add the Makefile target and link only `ds4_source_formats.o`.

**Kill gate:** STOP if the MXFP4 helper output does not match the
inline reference; investigate before touching anything else, because
the Sprint 007 fix would have regressed.

### Phase 2: Source-Layout Guard Regression Tests

**Files:**

- `tests/source_layout_guard_smoke.c` (create)
- `Makefile`

**Tasks:**

- [ ] Add `tests/source_layout_guard_smoke.c`. The test selects between:
      (a) preferred — an in-memory minimal `ds4_model`/`ds4_weights`
      fixture sufficient to make `model_uses_v100_source_layout` and
      `weights_validate_v100_source_layout` return true without the
      145 GiB model; or
      (b) fallback — a subprocess invocation of `./ds4` against a tiny
      committed synthetic source-layout fixture under
      `tests/test-vectors/source-layout-fixture/`. Prefer (a) and fall
      back to (b) only if (a) requires invasive surgery on `ds4.c`.
- [ ] Cases covered:
      1. `source_layout_oracle = false` -> open returns 1, stderr
         contains the Sprint 007 source-layout guard text;
      2. `source_layout_oracle = true` + `mtp_path != NULL` -> open
         returns 1, stderr contains "rejects MTP sidecars";
      3. `source_layout_oracle = true` + `backend != CPU` -> open
         returns 1, stderr contains "requires CPU backend";
      4. `source_layout_oracle = true` + `source_layout_oracle_sessions
         = false` -> `ds4_engine_open` succeeds, then `ds4_session_create`
         returns 1, stderr contains "diagnostic session unlock";
      5. `source_layout_oracle = true` + `source_layout_oracle_sessions
         = true` -> `ds4_engine_open` and `ds4_session_create` both
         succeed and the engine is marked oracle-only.
- [ ] Add the Makefile target.

**Kill gate:** STOP if implementing test (a) requires changing the
source-layout guard text or the order of the existing CLI checks. The
intent is to lock current behavior in, not to refactor it.

### Phase 3: V100 F16 KV Budget And Admission Surface

**Files:**

- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `tests/v100_context_smoke.c`
- (read-only) `ds4.c` for the head-dim constants

**Tasks:**

- [ ] In `ds4_v100_context.h`, add:
      ```c
      typedef enum {
          DS4_V100_LAYER_CLASS_SWA_ONLY = 0,
          DS4_V100_LAYER_CLASS_RATIO_4,
          DS4_V100_LAYER_CLASS_RATIO_128,
      } ds4_v100_layer_class;
      ```
      Add `layer_class` to `ds4_v100_layer_info`.
      Add `kv_ctx_tokens` and `kv_active_slots` to
      `ds4_v100_context_options`.
- [ ] Add a `ds4_v100_kv_class_bytes_per_slot(class, ctx_tokens, *out_bytes)`
      helper that returns nonzero on overflow. The bytes-per-slot
      formulas are derived from the architecture document; the helper
      lives in `ds4_v100_context.c` so the architecture math has one
      home.
- [ ] In `init_stage_map`, classify every layer 0..42 by the layer-class
      table and store on the layer info.
- [ ] In `validate_memory_budget`, when `opts.kv_ctx_tokens > 0` and
      `opts.kv_active_slots > 0`, compute per-stage planned KV bytes by
      summing per-layer class bytes across the stage's layer span and
      multiplying by `kv_active_slots`. Add an explicit overflow check.
      Write the result into `stages[i].planned_kv_bytes` before running
      the existing reserve check. Otherwise leave the caller-provided
      `planned_kv_bytes_per_gpu` as the source (back-compat for the
      Sprint 006 tests).
- [ ] Extend the KV_CACHE branch of `ds4_v100_classify_or_die` so it no
      longer hard-codes `EXEC_UNSUPPORTED`. When the source layout is
      F16 KV (the only supported KV format in this sprint), return
      `EXEC_DIAGNOSTIC_ONLY` with `forbidden_claim =
      "kv_population_in_sprint_008"`. The intent: classifying KV as
      diagnostic-only acknowledges that planning is wired but device
      population is still future work.
- [ ] In `tests/v100_context_smoke.c`:
      add `test_kv_admission_swa_only(...)`,
      `test_kv_admission_ratio_4(...)`,
      `test_kv_admission_ratio_128(...)`,
      `test_kv_admission_fails_closed_when_reserve_violated(...)`.
      Use `kv_ctx_tokens = 4096` and `kv_ctx_tokens = 262144` to keep
      both ends covered.
      Confirm classifier returns DIAGNOSTIC_ONLY for KV_CACHE with the
      F16 baseline and continues to return UNSUPPORTED for any other
      KV-format hint.
- [ ] Update `ds4_v100_context_print_report` to also emit
      `layer\tclass\tkv_bytes_per_slot` rows so the report has the
      admission breakdown for cluster logs.

**Kill gate:** STOP if the per-GPU F16 KV totals at any plausible
`kv_ctx_tokens` (e.g. 4K, 64K) push past the per-GPU reserve under the
documented layer-owned topology. That would mean the architecture
document's per-stage estimate has drifted and needs to be revisited
before implementing admission.

### Phase 4: Automated Official-Vector Oracle Runner

**Files:**

- `tests/source_oracle_runner.c` (create)
- `Makefile`

**Tasks:**

- [ ] Add `tests/source_oracle_runner.c`. The test:
      - reads `DS4_TEST_MODEL` env var; if unset and the default
        `/models/DSv4-Flash-256e-fixed.gguf` is not present, prints
        "skipped: source model not available" and returns 0;
      - reads `DS4_TEST_DS4_BIN` env var (default `./ds4`);
      - reads `DS4_TEST_CASE` env var (default `short_reasoning_plain`);
      - constructs prompt path
        `tests/test-vectors/prompts/<case>.txt`
        and official path
        `tests/test-vectors/official/<case>.official.json`;
      - `fork`/`execve`s the binary with the Sprint 007 incantation;
      - waits with a wall-clock cap (env `DS4_TEST_TIMEOUT_S`,
        default 1800);
      - parses the emitted `--dump-logprobs` JSON;
      - parses the official JSON;
      - asserts `selected.id` and `selected.text` equality;
      - emits a copy of the run JSON to
        `docs/sprints/drafts/SPRINT-008-cluster-logs/SPRINT-008-oracle-<case>.json`
        when the env var `DS4_TEST_ARCHIVE_DIR` is set.
- [ ] Use a tiny inline JSON reader; do not introduce a new dependency.
      The official files are small.
- [ ] Add `Makefile` target `tests/source_oracle_runner` and a phony
      `test-source-oracle-runner` that builds and runs it. Document in
      the rule that the runner exits 0 when the model is not present so
      laptop CI is not blocked.
- [ ] Do not modify the `ds4` binary's argument shape. The runner relies
      on the exact existing CLI surface so it cannot drift.

**Kill gate:** EXTEND if the runner works against `short_reasoning_plain`
on the cluster but cannot complete in the time budget; ship the runner
plus the host phases and record the timing in the report.

### Phase 5: CUDA BF16 Source-Row Device-Side Anchor

**Files:**

- `ds4_v100_context.h` (add probe declaration in the
  `#ifdef __cplusplus` block already present for the CUDA surfaces)
- `ds4_v100_context_cuda.cu` (add the probe + tiny BF16 -> F32 kernel)
- `tests/cuda_v100_kv_anchor_smoke.c` (create; despite the file name
  this is the BF16 source-row anchor — see Open Question 3 on naming)
- `Makefile`

**Tasks:**

- [ ] Declare:
      ```c
      int ds4_v100_cuda_bf16_source_row_probe(
              const uint16_t *host_bf16,
              uint64_t        n_elements,
              float          *host_f32_out,
              char           *err,
              size_t          errlen);
      ```
      The implementation:
      uploads `host_bf16` to device,
      launches a small kernel that computes
      `(float)((uint32_t)bf16 << 16)` reinterpreted as f32,
      copies the f32 device buffer back to `host_f32_out`,
      frees the device scratch,
      returns nonzero on any CUDA error.
- [ ] Add `tests/cuda_v100_kv_anchor_smoke.c`. The test:
      collects device facts via `ds4_v100_cuda_collect_device_facts`;
      if no V100 is present, skip with status code 0 and a clear log;
      builds a 1024-element BF16 buffer covering normal values,
      negative zero, NaN payload, denormals, and the four constants the
      Sprint 007 `source_dtypes_smoke` already exercises;
      calls the probe;
      computes the host expected output with
      `ds4_src_bf16_row_to_f32`;
      asserts byte-equality of every F32 word.
- [ ] Update the `Makefile` CUDA test rules. The host-only Darwin path
      prints "requires a CUDA build" so the laptop tree still builds.

**Kill gate:** EXTEND if the cluster build environment is unavailable
in the sprint window; archive the build instructions and ship the host
phases.

### Phase 6: Validation And Report

**Files:**

- `docs/sprints/SPRINT-008.md` (create on close)
- `docs/sprints/SPRINT-008-REPORT.md` (create on close)
- `docs/sprints/SPRINT-008-DEFERRED.md` (create)
- `docs/sprints/SPRINT-008-FOLLOWUPS.md` (create)
- `docs/sprints/VISION.md` (update Sprint 008 entry + pivot log row)
- `docs/sprints/drafts/SPRINT-008-mxfp4-parity.log`
- `docs/sprints/drafts/SPRINT-008-guard-smoke.log`
- `docs/sprints/drafts/SPRINT-008-context-smoke.log`
- `docs/sprints/drafts/SPRINT-008-cluster-logs/SPRINT-008-build.log`
- `docs/sprints/drafts/SPRINT-008-cluster-logs/SPRINT-008-oracle-short-reasoning-plain.json`
- `docs/sprints/drafts/SPRINT-008-cluster-logs/SPRINT-008-bf16-source-row-anchor.log`

**Tasks:**

- [ ] Run local: `make cpu tests/mxfp4_parity_smoke
      tests/source_layout_guard_smoke tests/v100_context_smoke
      tests/source_dtypes_smoke`, then the four host tests. Archive
      stdout/stderr.
- [ ] On the V100 pod with `CUDA_ARCH=sm_70`: `make clean &&
      make cpu tests/source_oracle_runner tests/cuda_v100_kv_anchor_smoke
      CUDA_ARCH=sm_70`; archive the build log.
- [ ] Run `DS4_TEST_MODEL=/models/DSv4-Flash-256e-fixed.gguf
      DS4_TEST_ARCHIVE_DIR=docs/sprints/drafts/SPRINT-008-cluster-logs
      ./tests/source_oracle_runner` on the pod; archive the run output
      and the official-vector comparison evidence JSON.
- [ ] Run `./tests/cuda_v100_kv_anchor_smoke` on the pod; archive the
      log.
- [ ] Run `./ds4 -m /models/DSv4-Flash-256e-fixed.gguf "hi"` (no
      `--source-layout-oracle`) and confirm the existing guard message
      and exit code 1; archive to
      `SPRINT-008-cluster-logs/SPRINT-008-source-normal-guard.log`.
- [ ] Write `SPRINT-008-REPORT.md` with the verdict, evidence pointers,
      what is *not* proven (prefill, decode throughput, indexer
      population, device KV residency, deployment), and the Sprint 009
      handoff.
- [ ] Update `docs/sprints/VISION.md` Sprint 008 entry and pivot log.

**Kill gate:** none — always runs.

---

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4_v100_context.h` | Modify | Add `ds4_v100_layer_class`, KV admission options, BF16 source-row probe declaration |
| `ds4_v100_context.c` | Modify | Layer-class classification, F16 KV per-slot helpers, KV admission step in `validate_memory_budget`, KV_CACHE classifier update |
| `ds4_v100_context_cuda.cu` | Modify | Add BF16 source-row probe + tiny kernel |
| `ds4.c` | Read-only | Source of head-dim/indexer-dim constants and the guard text checked by tests |
| `ds4_source_formats.[ch]` | Read-only | Source of host BF16/MXFP4 reference helpers |
| `tests/source_oracle_runner.c` | Create | Automated `--dump-logprobs` runner that asserts selected-token equality |
| `tests/source_layout_guard_smoke.c` | Create | Source-layout guard regression coverage |
| `tests/mxfp4_parity_smoke.c` | Create | Direct GGML `block_mxfp4` nibble-order parity test |
| `tests/v100_context_smoke.c` | Modify | KV admission cases and layer-class assertions |
| `tests/cuda_v100_kv_anchor_smoke.c` | Create | Cluster BF16 source-row device-side anchor |
| `Makefile` | Modify | New targets; cluster-only CUDA target stays in the existing Linux branch |
| `docs/sprints/SPRINT-008.md` | Create | Final merged sprint plan (after critique/merge cycle) |
| `docs/sprints/SPRINT-008-REPORT.md` | Create | Sprint verdict and evidence |
| `docs/sprints/SPRINT-008-DEFERRED.md` | Create | Items discussed and excluded from scope |
| `docs/sprints/SPRINT-008-FOLLOWUPS.md` | Create | Non-blocking items for Sprint 009 |
| `docs/sprints/VISION.md` | Modify | Sprint 008 outcome + pivot log row |
| `docs/sprints/drafts/SPRINT-008-cluster-logs/` | Create | Archived cluster artifacts |

---

## Definition Of Done

- [ ] `tests/mxfp4_parity_smoke` builds and passes locally and asserts
      `ds4_src_mxfp4_row_to_f32` matches the inline GGML `block_mxfp4`
      reference byte-for-byte over a representative row including
      negative zero, max positive, and max negative nibbles.
- [ ] `tests/source_layout_guard_smoke` builds and passes locally and
      covers normal-rejection, MTP-rejection, non-CPU-rejection, and
      session-without-diagnostic-unlock rejection paths.
- [ ] `ds4_v100_context.h` exposes `ds4_v100_layer_class`, `kv_ctx_tokens`,
      `kv_active_slots`, and per-class KV-bytes helpers.
- [ ] `ds4_v100_context.c` populates `layer_class` on every layer,
      computes per-stage planned KV bytes from the layer-class table
      when admission inputs are present, and fails closed on reserve
      violation with a precise message.
- [ ] `tests/v100_context_smoke` covers SWA-only, ratio-4, ratio-128
      class totals plus a reserve-violation case, and continues to pass
      all Sprint 006/007 cases.
- [ ] The KV_CACHE family classifier no longer hard-codes
      `EXEC_UNSUPPORTED`; it returns `EXEC_DIAGNOSTIC_ONLY` for the F16
      KV baseline and `EXEC_UNSUPPORTED` for any non-F16 KV hint.
- [ ] `tests/source_oracle_runner` builds locally and, when the model
      is unavailable, exits 0 with a clear "skipped" message.
- [ ] On the V100 pod, the runner asserts selected-token equality for
      `short_reasoning_plain` against the official fixture, and the
      cluster evidence JSON is archived under
      `docs/sprints/drafts/SPRINT-008-cluster-logs/`.
- [ ] `tests/cuda_v100_kv_anchor_smoke` builds with `CUDA_ARCH=sm_70`
      and, on the V100 pod, asserts bit-exact equality between the
      device BF16->F32 kernel output and `ds4_src_bf16_row_to_f32` over
      at least 1024 elements covering normal/negative-zero/NaN/denormal
      bit patterns.
- [ ] Normal source-layout open (no `--source-layout-oracle`) still
      fails with the existing guard message and exit code 1 on the
      cluster; archived.
- [ ] No persistent dequantized F16/F32 copy of any large source tensor
      is created; the BF16 probe's device F32 scratch is freed at end
      of test.
- [ ] No new host-backed, managed-memory, or SSD-backed runtime path is
      introduced. No production FP8/MXFP4 dense or grouped kernel is
      added. Prefill, indexer population, multi-token decode, MTP,
      server, tensor-parallel exceptions, and throughput benchmarks are
      not touched.
- [ ] `git diff --check` passes; the only untracked items committed are
      the new tests, the BF16 probe, the V100 KV surface, the Sprint 008
      docs, and the cluster log archive.
- [ ] `docs/sprints/SPRINT-008-REPORT.md` and `docs/sprints/VISION.md`
      are updated with the verdict, evidence, what was *not* proven,
      and the Sprint 009 handoff.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| In-memory minimal source-layout fixture for guard tests requires invasive surgery on `ds4.c` | Medium | Medium | Use subprocess fallback against a tiny committed synthetic source-layout GGUF fixture; do not refactor `ds4_engine_open` to make guard tests easier |
| Subprocess fallback flakes on the cluster | Low | Medium | Bound runner with `DS4_TEST_TIMEOUT_S`; emit "skipped" on timeout rather than failing |
| F16 KV per-slot totals at meaningful ctx push past the documented reserve under the layer-owned topology | Medium | High | Pick admission inputs that match the architecture document (single 4K-64K slot first); STOP and revisit the architecture estimate if the per-stage reserve is violated for a 4K slot |
| Adding `kv_ctx_tokens` admission breaks existing Sprint 006 callers | Low | Medium | Default behavior is unchanged when both `kv_ctx_tokens` and `kv_active_slots` are zero; existing tests stay green |
| BF16 device probe surfaces a CUDA build issue on the pod | Medium | Medium | The probe is intentionally tiny; if CUDA build is blocked, EXTEND with host phases shipped and record the blocker |
| `--dump-logprobs` JSON shape drifts between Sprint 007 and Sprint 008 | Low | Medium | Runner parser is tolerant on field ordering; the test pins on `steps[0].selected.id` and `steps[0].selected.text` which are stable |
| Sprint drifts into prefill, indexer population, or decode throughput | High | High | Each phase ends in a kill gate; KV admission is planning-only and never allocates device KV; the device anchor explicitly does not exercise a production kernel |
| The KV_CACHE classifier change accidentally unlocks `EXEC_DIAGNOSTIC_ONLY` for non-F16 KV hints | Medium | High | Tests assert both directions: F16 KV gets DIAGNOSTIC_ONLY, every other KV hint gets UNSUPPORTED |
| BF16 device probe's kernel produces different NaN payload than the host helper | Low | Medium | Probe uses `(uint32_t)bf16 << 16` reinterpret rather than `__bfloat162float` so NaN payloads remain identical to the host bit pattern; test covers a NaN payload case |
| Cluster archive directory grows large with per-run JSON | Low | Low | Runner overwrites a single fixed-name evidence file per case; rotation is deferred to a follow-up |

---

## Security

- The diagnostic unlock surface does not change. `--source-layout-oracle`
  remains CPU-only, source-layout-only, rejects MTP, and requires
  `--dump-logprobs` for sessions. The guard regression tests assert each
  rejection text exactly so log scrapers and CI checks continue to
  match.
- The runner shells out to `./ds4` with controlled arguments. The
  prompt path and official path are constructed from a closed allowlist
  of fixture case names; no caller-supplied path is concatenated into
  the argv. The `--dump-logprobs` output path is constructed under
  `/tmp` with a PID-tagged file name and opened by the engine, not by
  the test, so existing engine handling is reused.
- All model and pack bytes are treated as read-only by the host probe
  and by the device anchor. The device anchor allocates and frees its
  own scratch on each call; no resident F32 copy of any large source
  tensor is created.
- KV admission math validates overflow on every multiplication
  (per-layer bytes * stage layer count * active slots) and on the
  final per-stage total. Reserve checks remain fail-closed.
- The CUDA BF16 probe reads BF16 bytes and produces F32 in a bounded
  scratch buffer; the kernel does not touch any other tensor.
- The runner skips silently when the cluster model is unavailable so it
  is safe to invoke from a laptop pipeline; the skip message is
  non-actionable to a future user looking at archived logs.
- The Sprint 007 source-layout guard text and the oracle rejection
  texts are preserved verbatim so log-driven detection continues to
  work.

---

## Dependencies

- Sprint 007 source-layout oracle: `--source-layout-oracle`,
  `--dump-logprobs`, `source_layout_oracle_sessions`, and the
  `g_source_layout_oracle_f16_kv` global state.
- Sprint 007 source-format helpers (`ds4_source_formats.[ch]`) and the
  `tests/source_dtypes_smoke` reference.
- Sprint 006 V100 execution context (`ds4_v100_context.[ch]`,
  `ds4_v100_context_cuda.cu`), policy classifier, stage map, and
  descriptor-binding flow.
- Sprint 005 BF16 row-gather/expand bit-pattern contract used by the
  device anchor.
- `tests/test-vectors/prompts/short_reasoning_plain.txt` and
  `tests/test-vectors/official/short_reasoning_plain.official.json` as
  the SHIP fixture.
- Persistent source model at `/models/DSv4-Flash-256e-fixed.gguf` on the
  V100 cluster.
- 8x V100-SXM2-32GB pod `llamacpp-build-8gpu` in namespace `llm`, with
  CUDA 12.x toolchain and `sm_70`. Required for the device anchor and
  for the oracle runner; the host phases run on the laptop.
- Repo git rules: explicit `git add` paths only; do not add
  `logs/`.

---

## Open Questions

1. **Should the first device-side anchor be a KV-allocation/admission
   probe on the device, or a BF16 source-row read-and-convert probe?**
   This draft chooses the BF16 source-row probe because it reuses the
   Sprint 005 contract, requires no kernel scheduling, and is the
   smallest device-side proof that source bytes can be exercised
   end-to-end in Sprint 008. KV residency is treated as planning-only
   for now; device KV allocation belongs to Sprint 009 once prefill
   correctness is bounded.

2. **Is `--source-layout-oracle --dump-logprobs` a durable enough
   diagnostic unlock, or should Sprint 008 add the opaque code-level
   token originally sketched in Sprint 007?** This draft keeps the
   existing two-flag CLI unlock and locks it in with guard regression
   tests. Adding a second unlock token doubles the surface to audit and
   does not buy more safety than the current "must set both flags + CPU
   only + no MTP" combination.

3. **Should `tests/cuda_v100_kv_anchor_smoke.c` be renamed to
   `tests/cuda_bf16_source_row_anchor_smoke.c` since the actual probe
   is a BF16 source-row anchor, not a KV anchor?** Naming convergence
   is a merge-time decision. The intent's "first device-side anchor"
   language motivates the proposed file name (it is the device-side
   anchor for Sprint 008), but a precise name makes the test purpose
   clearer; recommend the precise name unless merge guidance disagrees.

4. **Which official vectors beyond `short_reasoning_plain` are cheap
   enough to run on the cluster during this sprint?**
   `short_italian_fact` is the shortest fixture after
   `short_reasoning_plain`. The runner makes the case name an env var
   so cluster time can decide. The SHIP bar requires only
   `short_reasoning_plain`; additional fixtures are EXTEND-grade
   evidence.

5. **Should the KV admission surface also validate F8 KV totals so
   Sprint 010 has a head start?** This draft says no. F8 KV is an
   explicit later-optimization gate. Modeling F8 totals now requires
   committing to an F8 KV byte-per-element value before any oracle
   exists, and would expand the architecture surface this sprint is
   trying to bound.

6. **Should the device anchor probe `token_embd.weight` from a real
   resident pack on the V100 pod, or always run on a synthetic BF16
   buffer?** This draft prefers the real resident path when the pack
   is loaded but accepts the synthetic fallback for the SHIP bar so
   the anchor never blocks on residency wiring that is not the
   purpose of this sprint. The cluster log records which mode ran.
