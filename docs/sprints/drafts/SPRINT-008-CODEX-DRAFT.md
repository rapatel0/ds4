# Sprint 008: Source Oracle Harness And V100 KV Admission Anchors

## Overview

Sprint 007 proved one narrow but important fact: the native DeepSeek V4 Flash
source layout can produce the correct first token in a guarded CPU oracle path,
and the project now has trustworthy BF16, F8_E4M3_B128, and MXFP4 source-format
helpers. Sprint 008 should turn that one-off proof into repeatable validation
and use it to shape the first production-relevant V100 surfaces without
pretending the appliance is ready for deployment, long-context serving, or
throughput work.

The sprint should stay bounded around three deliverables:

- a repo-native oracle harness that automates the existing official-vector
  check and hardens the source-layout guard behavior;
- a concrete F16 KV budget and admission surface in the V100 context layer for
  the layer-owned schedule, broken down by SWA-only, ratio-4/indexer, and
  ratio-128 layers;
- one conservative device-side source-format anchor that validates against the
  shared CPU oracle helpers without claiming a full production decode kernel.

This sprint does not unlock normal source-layout generation. It does not claim
full prompt prefill, full compressed-KV execution, multi-slot scheduling,
throughput readiness, server deployment, public oracle exposure, or broad V100
FP8/MXFP4 kernel coverage.

## Use Cases

1. **Automated official-vector oracle**: a maintainer can run one in-repo test
   command against `tests/test-vectors/official.vec` and verify that the source
   oracle still selects the expected first token without manually inspecting a
   JSON dump.
2. **Guard regression detection**: a developer can prove that normal
   source-layout open still fails closed, while the diagnostic oracle path
   remains CPU-only, MTP-rejecting, and session-gated.
3. **KV admission planning**: an operator can open the V100 context with a real
   `pack-index.tsv`, a target context size, and a slot count, then get a
   deterministic per-stage F16 KV budget/admission report before runtime KV
   allocation exists.
4. **Layer-class visibility**: the next sprint can inspect which layers are
   SWA-only, ratio-4 with indexer state, or ratio-128 compressed-only, and tie
   each class to expected raw/compressed/indexer bytes.
5. **Device-side anchor parity**: a CUDA developer can run one bounded packed
   source-format probe on synthetic rows and compare the result directly against
   the shared CPU source-format helpers.

## Architecture

### 1. Oracle Harness, Not Oracle Widening

Sprint 008 should keep the current runtime boundary: normal source-layout
generation still fails closed, and the oracle remains diagnostic-only. The new
work is automation and guard hardening, not a broader unlock.

The most direct repo-native path is to extend `tests/ds4_test.c` with a
dedicated source-oracle vector mode, for example `--source-logprob-vectors`,
that:

- opens the source model with `backend=CPU`,
  `source_layout_oracle=true`, and the explicit session unlock;
- reuses the existing `official.vec` parser and official top-logprob fixtures;
- verifies at minimum the selected first token for
  `short_reasoning_plain`, with optional top-logprob checks reused from the
  existing vector comparison logic;
- keeps the current CLI diagnostic mode available, but does not make CLI JSON
  dumps the primary validation surface.

Guard hardening should stay close to the engine boundary in `ds4.c` and tests:

- ordinary source-layout open still rejects runtime use;
- non-CPU oracle attempts fail closed;
- MTP remains rejected;
- source-oracle sessions still require the explicit diagnostic session gate;
- no new public serving or chat path is added for the oracle.

### 2. Shared KV Budget Math, Not Full KV Execution

Sprint 008 should expose the first V100 KV planning contract without claiming
that prefill or decode is solved. The context surface in
`ds4_v100_context.[ch]` should grow from a single scalar
`planned_kv_bytes_per_gpu` into a derived F16 KV plan tied to actual DS4 layer
classes.

The source of truth should be the existing DS4 cache math already present in
`ds4.c`:

- `ds4_layer_compress_ratio()` for SWA-only / ratio-4 / ratio-128 layer
  classes;
- `layer_attn_state_bytes()` and `layer_index_state_bytes()` for recurrent
  state sizes;
- the existing raw/compressed/indexer cache accounting used by session payload
  and graph memory estimation.

If extracting these helpers is cleaner than duplicating them, Sprint 008 should
promote only the minimal byte-count logic into a small shared internal helper
module and keep session execution code in `ds4.c`.

The V100 context should then expose:

- per-layer class metadata;
- per-stage planned raw/compressed/indexer F16 KV bytes;
- an admission function that answers whether `(ctx_size, active_slots)` fits
  within reserve for the current topology;
- report output that makes the KV plan auditable in text logs.

This is a planning and validation surface only. It does not allocate full
device KV buffers, append prompt tokens, run sparse attention, or claim
long-context throughput.

### 3. One Conservative Device-Side Source Anchor

Sprint 008 should include exactly one bounded device-side source-format anchor.
The best first anchor is a synthetic `F8_E4M3_B128` row decode or row-dot probe
that mirrors the existing `tests/cuda_bf16_probe.c` pattern:

- upload a small packed synthetic row into a `ds4_gpu_arena`;
- run one CUDA helper that consumes the packed row directly;
- compare the output against `ds4_src_f8_e4m3_b128_row_to_f32()` or
  `ds4_src_f8_e4m3_b128_row_dot()`;
- fail closed on malformed block spans, row bounds, or undersized outputs.

That anchor is production-relevant because dense attention and output-path
families depend on the blocked FP8 layout, but it is still narrow enough to
avoid pretending that full dense GEMMs or routed-expert execution are ready.

MXFP4 should still receive hardening this sprint, but primarily through parity
tests and guardrails:

- add a direct regression in `tests/source_dtypes_smoke.c` for the GGML
  `block_mxfp4` low-half/high-half nibble layout;
- use that parity result as the reference for later routed-expert kernel work;
- do not require a full device-side MXFP4 expert kernel to ship Sprint 008.

### 4. Validation Flow

```text
tests/test-vectors/official.vec
    |
    v
source-oracle ds4_test mode
    |
    +--> first-token / top-logprob validation
    +--> source-layout guard regression coverage

ds4.c KV byte helpers + layer ratio map
    |
    v
ds4_v100_context KV planner
    |
    +--> per-layer class report
    +--> per-stage KV bytes
    +--> admission verdict for ctx/slots

shared CPU source-format helpers
    |
    v
bounded CUDA packed-row probe
    |
    +--> parity against CPU helper
    +--> no claim of full decode or throughput
```

## Implementation

### Phase 1: Oracle Automation And Guard Hardening

**Files:**
- `ds4.c`
- `ds4.h`
- `tests/ds4_test.c`
- `tests/test-vectors/README.md`
- `Makefile`

**Tasks:**
- [ ] Add a dedicated source-oracle official-vector test mode in
      `tests/ds4_test.c` that opens the source model in the guarded CPU oracle
      configuration and checks at least the selected first token from
      `official.vec`.
- [ ] Reuse the existing vector parser and top-logprob comparison machinery
      rather than creating a separate fixture format.
- [ ] Add targeted guard checks for normal source-layout rejection, non-CPU
      rejection, MTP rejection, and missing session unlock rejection.
- [ ] Keep the CLI diagnostic path narrow; do not add a public normal-user
      source-layout execution flag.
- [ ] Document the new repo-native validation command in
      `tests/test-vectors/README.md`.

### Phase 2: Shared KV Accounting Surface

**Files:**
- `ds4.c`
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `tests/v100_context_smoke.c`
- `tests/cuda_v100_context_smoke.c`

**Tasks:**
- [ ] Promote or share the minimal DS4 layer-ratio and KV byte-count formulas
      needed for planning, while leaving execution ownership in `ds4.c`.
- [ ] Extend the V100 context model with layer-class metadata for SWA-only,
      ratio-4/indexer, and ratio-128 layers.
- [ ] Replace the current coarse per-GPU `planned_kv_bytes_per_gpu` story with
      a derived F16 KV report that shows raw, compressed, and indexer bytes.
- [ ] Add an admission/check surface for `(ctx_size, active_slots)` against the
      current reserve policy and stage ownership.
- [ ] Extend local and CUDA context smokes to assert the expected layer-class
      counts and to fail when the requested KV plan exceeds reserve.

### Phase 3: Context Reporting And Executable Planning

**Files:**
- `ds4_v100_context.h`
- `ds4_v100_context.c`
- `tests/v100_context_smoke.c`
- `tests/cuda_v100_context_smoke.c`

**Tasks:**
- [ ] Print per-stage planned KV bytes in a stable report format suitable for
      archiving under `docs/sprints/drafts/SPRINT-008-*`.
- [ ] Make the smoke tests accept explicit context-size and slot-count inputs so
      the admission result is reproducible from the command line.
- [ ] Keep the default report conservative: F16 KV baseline only, with no F8 KV
      readiness claim.
- [ ] Stop the sprint if the KV planner requires hidden assumptions about
      runtime allocation behavior that are not already encoded in the repo.

### Phase 4: Conservative CUDA Source Anchor

**Files:**
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tests/cuda_source_dtypes_smoke.c`
- `tests/source_dtypes_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add one packed FP8 source-row device probe API beside the existing BF16
      arena probe surface.
- [ ] Build a synthetic CUDA smoke that uploads one or two
      `F8_E4M3_B128` rows, runs the bounded device helper, and compares against
      the shared CPU source-format helpers.
- [ ] Preserve the Sprint 007 MXFP4 correction with a direct nibble-layout
      regression in `tests/source_dtypes_smoke.c`.
- [ ] Keep the device anchor synthetic and bounded; do not widen into full
      production GEMMs, routed-expert kernels, or output-head decode.

### Phase 5: Evidence, Report, And Close-Out

**Files:**
- `docs/sprints/drafts/SPRINT-008-ORACLE.log`
- `docs/sprints/drafts/SPRINT-008-GUARD.log`
- `docs/sprints/drafts/SPRINT-008-KV-ADMISSION.log`
- `docs/sprints/drafts/SPRINT-008-CUDA-SOURCE-ANCHOR.log`
- `docs/sprints/SPRINT-008-REPORT.md`
- `docs/sprints/SPRINT-008-FOLLOWUPS.md`

**Tasks:**
- [ ] Archive local validation output for the oracle runner, guard checks, KV
      admission report, and CUDA source anchor.
- [ ] If cluster validation is used, archive only the bounded oracle and CUDA
      smoke evidence needed for the sprint verdict.
- [ ] Write the Sprint 008 report so it states exactly what is proven:
      automated oracle parity, F16 KV admission accounting, and one bounded
      device-side source-format anchor.
- [ ] Record remaining gaps explicitly instead of implying deployment or
      throughput readiness.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `ds4.c` | Modify | Reuse or expose minimal KV accounting helpers and keep source-oracle guard behavior fail-closed. |
| `ds4.h` | Modify | Document any narrow engine/test options needed for guarded oracle validation. |
| `ds4_v100_context.h` | Modify | Add per-layer/per-stage KV planning and admission surfaces. |
| `ds4_v100_context.c` | Modify | Implement F16 KV accounting, layer-class reporting, and reserve-based admission checks. |
| `ds4_gpu.h` | Modify | Expose one bounded packed-source CUDA probe API beside the existing BF16 arena probe surface. |
| `ds4_cuda.cu` | Modify | Implement the conservative device-side FP8 packed-row anchor used only for bounded validation. |
| `tests/ds4_test.c` | Modify | Add automated source-oracle official-vector and guard regression coverage. |
| `tests/v100_context_smoke.c` | Modify | Validate layer classes, KV accounting, and admission behavior locally. |
| `tests/cuda_v100_context_smoke.c` | Modify | Validate the same V100 context admission surface against visible CUDA topology. |
| `tests/cuda_source_dtypes_smoke.c` | Create | Validate the bounded packed-source CUDA anchor against the shared CPU source-format helpers. |
| `tests/source_dtypes_smoke.c` | Modify | Add direct MXFP4 layout hardening and any additional packed-source parity checks. |
| `tests/test-vectors/README.md` | Modify | Document the source-oracle vector runner and expected validation workflow. |
| `Makefile` | Modify | Build the new/updated test targets and keep validation executable in-repo. |
| `docs/sprints/SPRINT-008-REPORT.md` | Create during execution | Record verdict, evidence, and exact remaining gaps. |
| `docs/sprints/SPRINT-008-FOLLOWUPS.md` | Create during execution if needed | Capture post-sprint hardening and next-step items. |

## Definition of Done

- `make cpu ds4_test tests/source_dtypes_smoke tests/v100_context_smoke`
  succeeds locally.
- `./tests/source_dtypes_smoke` passes, including the direct MXFP4 layout
  regression.
- `./tests/v100_context_smoke` passes with at least one explicit KV admission
  case and one over-budget rejection case.
- `./ds4_test --source-logprob-vectors` (or the accepted equivalent test mode)
  runs against the source model and verifies the expected first token for
  `short_reasoning_plain` without manual JSON inspection.
- A bounded source-oracle guard test proves that ordinary source-layout runtime
  open still fails closed.
- If CUDA is available, `make tests/cuda_source_dtypes_smoke` succeeds and the
  new device-side packed-source probe passes locally or on the V100 cluster
  with archived output.
- `git diff --check` passes.
- `docs/sprints/SPRINT-008-REPORT.md` and any follow-up file clearly separate
  what Sprint 008 proved from what remains deferred.

## Risks

- **Scope creep into full prefill**: once KV byte accounting exists, it will be
  tempting to start wiring real prompt prefill or sparse attention. The sprint
  should stop at planning, admission, and bounded anchors unless a small
  addition is clearly required for validation.
- **Formula drift between `ds4.c` and V100 planning code**: duplicating the
  DS4 layer-ratio or KV byte math would make the planner untrustworthy. The
  mitigation is to share the smallest possible helper surface or prove parity
  in tests.
- **Oracle automation that bypasses guards indirectly**: a convenient test
  harness must not become a hidden runtime unlock. Keep the automation inside
  `ds4_test` or dedicated diagnostics and continue rejecting normal runtime use.
- **False confidence from one CUDA anchor**: one packed-row probe does not
  prove decode correctness. The report must state that it is only a bounded
  device-side source-format anchor.
- **Reserve estimates may be too optimistic**: if admission math ignores
  scratch, relay, or output-head pressure, the planner will over-admit. The
  sprint should treat missing accounting as a blocker, not as a reason to guess.

## Security

- Keep the source-layout oracle diagnostic-only, CPU-only, and session-gated.
- Do not add a public CLI/server mode that exposes source-layout generation as
  a supported path.
- Preserve fail-closed behavior for unsupported source dtypes, malformed packed
  rows, out-of-bounds probe requests, and over-budget KV admission attempts.
- Keep validation local to repo artifacts and existing model fixtures; Sprint
  008 does not require new network-facing surfaces or credential-bearing
  services.

## Dependencies

- `docs/sprints/VISION.md`
- `docs/architecture/DS4-V100-LAYOUT.md`
- `docs/sprints/SPRINT-007-REPORT.md`
- `docs/sprints/SPRINT-007-DEFERRED.md`
- `docs/sprints/SPRINT-007-FOLLOWUPS.md`
- Existing source-format helpers in `ds4_source_formats.[ch]`
- Existing official-vector fixtures in `tests/test-vectors/`
- Existing V100 context and CUDA smoke surfaces in
  `ds4_v100_context.[ch]`, `ds4_v100_context_cuda.cu`,
  `tests/v100_context_smoke.c`, and `tests/cuda_v100_context_smoke.c`
- A CUDA build and visible GPU only for the bounded device-side anchor; the
  oracle automation and KV planner must still be useful without cluster access

## Open Questions

1. Should the source-oracle automation live entirely in `ds4_test`, or does the
   project still want a thin wrapper around the existing `--dump-logprobs`
   diagnostic command for operator workflows?
2. Is it cleaner to extract DS4 KV byte formulas into a tiny shared internal
   module, or to keep the V100 planner local and assert parity with `ds4.c` in
   tests?
3. Is a synthetic `F8_E4M3_B128` row probe the right first device-side anchor,
   or is there a narrow real-pack row read that gives better evidence without
   dragging the sprint into broader runtime work?
4. Which additional official vectors, if any, are cheap enough to add after
   `short_reasoning_plain` without turning oracle validation into a long-running
   cluster task?
