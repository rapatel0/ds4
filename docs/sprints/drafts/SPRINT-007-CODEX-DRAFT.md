# Sprint 007: Guarded Source-Layout Single-Slot Decode Oracle

## Overview

Sprint 006 shipped the V100 execution context, descriptor policy, and fail-closed
guardrails needed to talk about source-layout execution seriously. What is still
missing is a trustworthy correctness path for the native DeepSeek V4 Flash
source GGUF: the current CPU reference helpers and session decode flow are still
biased toward the legacy F16/Q8_0 runtime layout, while `ds4_engine_open()`
correctly rejects normal source-layout generation.

Sprint 007 should close that gap with a guarded, CPU-only source oracle. The
goal is not to make V100 production decode succeed yet, and it is not to add
fake broad FP32 fallback. The goal is to teach the reference path the exact
BF16, F32, F8_E4M3_B128, and MXFP4 semantics of the source model well enough to
run one-slot, small-context decode on the native source GGUF and compare the
result against trusted official vectors.

The implementation should extend the existing DS4 seams instead of creating a
parallel runtime. Exact source-dtype decode helpers should be promoted into a
shared helper surface, `ds4.c` should gain source-layout-aware reference
matvec/embedding/output dispatch, and `ds4_test` should get a named oracle mode
for official-vector comparison. Normal source-layout runtime startup must remain
guarded everywhere outside that diagnostic path.

## Use Cases

1. **Model-less dtype proof**: a developer can run synthetic BF16, FP8, and MXFP4
   fixtures locally and verify exact decode semantics before touching full-model
   execution.
2. **Guarded source oracle**: a maintainer can open the native source GGUF in a
   named CPU-only diagnostic mode and evaluate a short prompt without widening
   normal runtime support.
3. **Official-vector regression**: an operator can run a bounded
   source-layout-vs-official comparison on the cluster and archive token/top-logprob
   evidence for Sprint 007.
4. **Future kernel anchor**: later CUDA/V100 kernel work can reuse the exact
   source-format semantics and fail-closed tensor dispatch instead of re-deriving
   FP8 and MXFP4 behavior from scratch.
5. **Guard regression detection**: the project can prove that ordinary
   `ds4_engine_open()` still rejects source-layout generation unless the explicit
   oracle path is selected.

## Architecture

Sprint 007 should introduce one narrow new concept: a source-oracle execution
mode that reuses the existing CPU reference/session flow but only after exact
source-format decode semantics are available.

```text
source GGUF
    |
    v
weights_bind + source-layout validation
    |
    +--> ordinary engine open
    |        |
    |        +--> still fails closed for source layout
    |
    +--> source oracle open (CPU-only, explicit opt-in)
             |
             v
      source-aware reference helpers
      - BF16 row decode
      - F32 control matvec
      - FP8 blocked dense matvec
      - MXFP4 grouped expert matvec
             |
             v
      existing session / eval flow
             |
             v
      ds4_test source-vector comparison
             |
             +--> selected token equality
             +--> official top-logprob membership
             +--> optional full-logit oracle when available
```

### Shared Source-Dtype Contract

The exact source-format decode rules should live in one reusable helper surface,
not as duplicated static snippets. `gguf-tools/quants.[ch]` already owns BF16 and
F16 conversion helpers and is the right place to expose:

- E8M0 block-scale decode used by the source FP8 and MXFP4 tensors;
- E4M3fn scalar decode for `F8_E4M3_B128`;
- MXFP4 nibble decode and per-32-value scale application;
- small bounded helpers for row or block decode that the CPU reference path can
  use without materializing persistent expanded weights.

`gguf-tools/deepseek4-quantize.c` should consume the same shared helpers so
there is one canonical interpretation of source bytes across the quantizer and
the runtime oracle.

### Reference Decode Boundary

`ds4.c` should remain the reference execution owner. Sprint 007 should add
source-layout-aware CPU helpers rather than inventing a second interpreter.
Concretely:

- embeddings and BF16 dense tensors should stop assuming IEEE F16 storage;
- HC control, router, norms, and other F32 control tensors should use an
  explicit F32 matvec path;
- source FP8 dense tensors should accumulate with block-scale application in the
  reference path;
- routed expert execution should use exact MXFP4 decode in a grouped expert
  reference path;
- all source-layout dispatch must fail closed on unsupported type, unexpected
  block shape, or mixed semantics.

The oracle path should dequantize only into bounded local scratch during a
single matvec or grouped expert operation. No persistent F16/F32 mirrors of
large source tensors should be introduced.

### Guard Boundary

The normal source-layout rejection in `ds4_engine_open()` stays intact. The only
new bypass should be an explicit oracle mode in `ds4_engine_options`, limited to
the CPU backend and intended for tests/diagnostics. It must log clearly that it
is a correctness oracle, not a supported V100 runtime path.

The oracle mode should also stay narrow:

- no MTP;
- no multi-slot or throughput claims;
- no server exposure;
- no host-backed or offloaded success path;
- no silent fallback to "good enough" approximations when FP8 or MXFP4 semantics
  are uncertain.

## Implementation

### Phase 1: Shared Source-Format Helpers (~20% of effort)

**Files:**
- `gguf-tools/quants.h`
- `gguf-tools/quants.c`
- `gguf-tools/deepseek4-quantize.c`
- `Makefile`

**Tasks:**
- [ ] Promote exact E8M0, E4M3fn, and MXFP4 decode helpers into
      `gguf-tools/quants.[ch]` beside the existing BF16/F16 conversion surface.
- [ ] Add small reusable helpers for bounded FP8 block decode and MXFP4
      per-32-value decode so the runtime oracle can use the same semantics as
      the quantizer.
- [ ] Switch `gguf-tools/deepseek4-quantize.c` to the shared helpers so source
      dtype interpretation is canonical in one place.
- [ ] Update the build so core/runtime and test targets can link the shared
      helper object without dragging in unrelated quantization entry points.

### Phase 2: Source-Layout CPU Reference Dispatch (~35% of effort)

**Files:**
- `ds4.c`
- `ds4.h`

**Tasks:**
- [ ] Add explicit source-layout reference helpers for F32 control matvec,
      BF16 dense matvec, FP8 blocked dense matvec, and MXFP4 grouped expert
      matvec.
- [ ] Replace or wrap `embed_token_f16()` so source BF16 embeddings are decoded
      by BF16 semantics instead of F16 semantics.
- [ ] Make output projection, HC control, router control, compressor/indexer
      BF16 families, and other source-layout call sites route through
      dtype-aware helpers instead of hard-coded F16/Q8_0 assumptions.
- [ ] Keep dequant bounded to local scratch or direct accumulation; do not
      persist expanded large source weights.
- [ ] Fail closed when a source-layout tensor reaches an unsupported legacy
      helper or when source block metadata is inconsistent with expected layout.

### Phase 3: Guarded Source-Oracle Mode (~15% of effort)

**Files:**
- `ds4.h`
- `ds4.c`
- `tests/ds4_test.c`

**Tasks:**
- [ ] Add an explicit source-oracle flag or mode to `ds4_engine_options`.
- [ ] Permit source-layout engine open only when the mode is set and the backend
      is CPU; keep ordinary `ds4_engine_open()` behavior unchanged otherwise.
- [ ] Emit a clear startup banner explaining that the engine is running in a
      guarded source-oracle mode, not a supported production runtime.
- [ ] Keep the oracle scope narrow: no MTP, no server, no throughput-oriented
      session use, and no hidden widening of the V100 precision policy.

### Phase 4: Synthetic Tests And Official-Vector Checks (~20% of effort)

**Files:**
- `tests/source_dtypes_smoke.c`
- `tests/ds4_test.c`
- `Makefile`

**Tasks:**
- [ ] Add a model-less smoke target for BF16, E8M0, E4M3fn, FP8 blocked rows,
      and MXFP4 grouped expert decode with exact expected values.
- [ ] Add failure tests for bad block shapes, unsupported source dtypes, and
      attempts to use source-layout execution outside oracle mode.
- [ ] Add a new `ds4_test` mode such as `--source-logprob-vectors` that reuses
      the existing `official.vec` parser but opens the source model in oracle
      mode.
- [ ] Preserve the current `--logprob-vectors` behavior for supported legacy
      runtime layouts.
- [ ] Allow optional full-logit comparison when `DS4_ORACLE_LOGITS` is provided,
      but keep selected-token and official top-logprob membership as the minimum
      comparison contract.

### Phase 5: Cluster Proof And Close-Out (~10% of effort)

**Files:**
- `docs/sprints/drafts/SPRINT-007-DTYPE-SMOKE.log`
- `docs/sprints/drafts/SPRINT-007-SOURCE-ORACLE.log`
- `docs/sprints/drafts/SPRINT-007-GUARD.log`
- `docs/sprints/SPRINT-007-REPORT.md`

**Tasks:**
- [ ] Build the CPU path and new smoke targets on Linux with the current source
      model workflow.
- [ ] Run the model-less dtype smoke locally and archive the exact output.
- [ ] Run the bounded source-oracle vector check against
      `/models/DSv4-Flash-256e-fixed.gguf` on the cluster and archive stdout/stderr.
- [ ] Re-run the ordinary source-layout open path without oracle mode and
      archive the guard failure output.
- [ ] If exact FP8 or MXFP4 semantics remain unresolved, stop after the helper
      and test work, record the blocker in `SPRINT-007-REPORT.md`, and do not
      claim shipped decode correctness.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `gguf-tools/quants.h` | Modify | Export shared source-format decode helpers needed by both the quantizer and the runtime oracle. |
| `gguf-tools/quants.c` | Modify | Implement canonical E8M0, E4M3fn, and MXFP4 decode logic plus bounded helper routines. |
| `gguf-tools/deepseek4-quantize.c` | Modify | Reuse the shared helper surface so source-dtype interpretation stays consistent. |
| `ds4.h` | Modify | Add the explicit source-oracle engine option and document its narrow scope. |
| `ds4.c` | Modify | Add source-layout-aware reference decode helpers and guard-preserving oracle entry behavior. |
| `tests/source_dtypes_smoke.c` | Create | Model-less exact-value tests for BF16, FP8, and MXFP4 source semantics. |
| `tests/ds4_test.c` | Modify | Add guarded source-oracle vector comparison mode and guard regression coverage. |
| `Makefile` | Modify | Build the shared helper object, new smoke target, and any updated test wiring. |
| `docs/sprints/drafts/SPRINT-007-DTYPE-SMOKE.log` | Create | Archive model-less source-dtype smoke output. |
| `docs/sprints/drafts/SPRINT-007-SOURCE-ORACLE.log` | Create | Archive bounded official-vector comparison output from the source model. |
| `docs/sprints/drafts/SPRINT-007-GUARD.log` | Create | Archive proof that normal source-layout open still fails closed. |
| `docs/sprints/SPRINT-007-REPORT.md` | Create | Record the final `SHIP`, `EXTEND`, or `STOP` verdict and evidence. |

## Definition of Done

- [ ] One shared helper surface defines the exact BF16, E8M0, E4M3fn, and MXFP4
      decode semantics used by both the runtime oracle and the quantizer.
- [ ] Source-layout CPU reference decode no longer interprets BF16 embeddings or
      output weights as F16, and no longer relies on legacy Q8_0-only dispatch
      for source FP8/MXFP4 tensors.
- [ ] A guarded source-oracle mode exists and is the only path that may bypass
      the ordinary source-layout rejection.
- [ ] Model-less dtype smoke tests pass for representative BF16, FP8, and MXFP4
      fixtures plus failure cases.
- [ ] At least one bounded official-vector case runs against the native source
      GGUF and matches selected-token behavior plus official top-logprob
      membership for each checked step.
- [ ] The project archives source-oracle, dtype-smoke, and guard logs under
      `docs/sprints/drafts/`.
- [ ] No persistent dequantized F16/F32 copies of large source tensors are left
      resident as part of the oracle path.
- [ ] Normal `ds4_engine_open()` still rejects source-layout generation when the
      oracle mode is not explicitly selected.
- [ ] If exact FP8 or MXFP4 semantics cannot be established, Sprint 007 exits
      with `STOP` and a blocker report instead of weakening the guardrails.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| The shipped source GGUF uses FP8 block-scale semantics that differ from the current quantizer assumptions. | High | High | Extract one canonical helper surface first, validate it with synthetic fixtures, and stop before decode if semantics remain ambiguous. |
| MXFP4 routed-expert layout or nibble decode assumptions are incomplete for exact reference math. | High | High | Add focused model-less MXFP4 tests and keep a hard stop gate before claiming one-token decode. |
| The sprint expands into a second runtime instead of a narrow correctness oracle. | Medium | High | Reuse `ds4.c` and `ds4_test`, keep CPU-only scope, and forbid throughput/server/MTP work in this sprint. |
| The new oracle mode accidentally weakens the normal source-layout guard. | Medium | High | Make the bypass explicit in `ds4_engine_options`, test the failure case separately, and keep default behavior unchanged. |
| CPU source-model verification on the cluster is slower than expected. | Medium | Medium | Limit validation to short official-vector cases, allow filtered case selection, and archive only the bounded proof run. |

## Security

- Source-oracle mode must stay opt-in and diagnostic only; it should not be
  exposed as a default server or general CLI runtime path.
- The sprint must not add host-backed, managed-memory, SSD-backed, or persistent
  dequantized success paths for source-layout execution.
- Official-vector validation should remain deterministic and bounded; it should
  not execute model-generated tools or widen request-surface behavior.
- Any optional full-logit oracle input should be read from an explicit local
  file path and treated as diagnostic evidence, not as a silent behavior change.

## Dependencies

- `docs/sprints/VISION.md` for sprint sequencing and the post-Sprint-006
  correctness goal.
- `docs/architecture/DS4-V100-LAYOUT.md` for precision policy, source dtype
  mapping, and the "no persistent dequantized weights" rule.
- Sprint 006 context/guard work in `ds4_v100_context.[ch]` and
  `docs/sprints/SPRINT-006-REPORT.md`, which established the current fail-closed
  source-layout boundary.
- Existing official vector fixtures under `tests/test-vectors/`, especially
  `tests/test-vectors/official.vec`.
- The native source GGUF at `/models/DSv4-Flash-256e-fixed.gguf` for bounded
  cluster validation.
- Linux CPU build support for the real source-model oracle run; CUDA kernels are
  not a dependency for the core Sprint 007 deliverable.

## Open Questions

1. Are the FP8 and MXFP4 decode rules currently embedded in
   `gguf-tools/deepseek4-quantize.c` fully authoritative for the source GGUF we
   ship against, or do we need an additional extracted fixture from the exact
   model bytes before calling the oracle exact?
2. Is selected-token equality plus official top-logprob membership sufficient to
   mark `SHIP`, or should Sprint 007 also require a separately captured
   full-logit oracle for at least one case?
3. What hard cap on prompt length and decode steps keeps the cluster CPU oracle
   practical enough to rerun during regression checks?
4. Should the source-oracle mode remain test-only in `ds4_test`, or is a narrow
   inspect-only CLI hook worth adding for manual diagnosis after the sprint
   ships?
