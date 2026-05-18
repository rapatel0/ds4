---
sprint: 007
title: Source-Layout Single-Slot Decode Oracle
status: completed
verdict: SHIP
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
intent: drafts/SPRINT-007-INTENT.md
merge_notes: drafts/SPRINT-007-MERGE-NOTES.md
deferred: SPRINT-007-DEFERRED.md
report: SPRINT-007-REPORT.md
---

# SPRINT-007: Source-Layout Single-Slot Decode Oracle

## Overview

Sprint 006 shipped the V100 execution context, descriptor policy, memory and
topology checks, HC relay smoke, and no-math layer skeleton. The source GGUF is
recognized and validated, but normal generation still fails closed because the
runtime does not yet have correct source-format execution for BF16/F32,
F8_E4M3_B128, and MXFP4 tensors.

Sprint 007 is the next correctness gate. It introduces a guarded, CPU-only
source-layout oracle that can prove exact source dtype semantics, correct the
reference dispatch places that currently assume legacy F16/Q8_0 tensors, and
compare one small official prompt against the existing DeepSeek V4 Flash
fixtures. This is not a V100 production kernel sprint and not a broad FP32
runtime fallback.

The V100 precision policy remains unchanged:

- BF16 source tensors are converted explicitly for reference/oracle work or
  later packed as FP16/F32 control inputs; V100 does not execute native BF16
  tensor-core GEMMs.
- F8_E4M3_B128 and MXFP4 are packed source/runtime inputs. The oracle computes
  them in FP32 for correctness only; later production kernels must unpack or
  dequant into FP16 HMMA tiles or validated low-bit/integer kernels.
- FP32 is acceptable for control, reductions, diagnostics, and the CPU oracle.
  It is not the production model-math default.
- Normal source-layout generation remains guarded unless an explicit oracle
  option is set by a test or dedicated diagnostic tool.

## Execution Result

Sprint 007 shipped as `SHIP`.

The implementation added shared source-format helpers, source-aware CPU
reference dispatch for the diagnostic path, a narrow source-layout oracle
session gate, and cluster evidence that the official `short_reasoning_plain`
fixture selects the expected first token exactly. During validation the oracle
exposed an MXFP4 block-layout mismatch; the source helper now matches GGML's
`block_mxfp4` nibble order. It also confirmed that source-layout correctness
starts from F16 KV, with F8 KV left as a later optimization gate.

Normal source-layout generation still fails closed. The successful path is a
bounded diagnostic `--dump-logprobs` oracle run, not a deployed serving path and
not a production V100 kernel path.

The full verdict, evidence, deviations, and Sprint 008 handoff are recorded in
`docs/sprints/SPRINT-007-REPORT.md`.

## Outcome Contract

- `SHIP`: exact source-format helpers exist and pass synthetic tests; source
  BF16/F32/FP8/MXFP4 dispatch is guarded and fail-closed; a CPU-only oracle
  path opens the source model only through an explicit diagnostic option; at
  least one short official vector matches the first generated token exactly on
  the cluster; normal source-layout generation still fails closed; logs and the
  report are archived.
- `EXTEND`: source helpers, guard plumbing, and synthetic tests land, but full
  first-token oracle comparison is blocked by cluster access, unacceptable
  runtime, or a top-K-only match. The exact blocker and completed primitives are
  recorded without weakening the guard.
- `STOP`: F8_E4M3_B128 or MXFP4 semantics cannot be established against the
  in-tree dequant reference; the oracle bypass cannot be kept narrow; or the
  implementation starts drifting into prefill, KV, MTP, server, throughput, or
  production CUDA kernels.

## Non-Goals

- No normal source-model generation unlock.
- No V100 production FP8/MXFP4/INT/FP16-HMMA kernel implementation.
- No prefill, compressed KV, indexer-state growth, long-context work, or
  multi-token prompt processing beyond the bounded oracle fixture.
- No multi-slot scheduling, wavefront scheduling, batching, MTP, or speculative
  decode.
- No server/API deployment, health checks, or throughput benchmark.
- No tensor-parallel exceptions.
- No persistent dequantized F16/F32 copies of large source tensors.
- No host-backed, SSD-backed, managed-memory, or offloaded successful production
  path.

## Planning Inputs

| File | Role |
|---|---|
| `docs/sprints/VISION.md` | Sprint sequence and North Star |
| `docs/architecture/DS4-V100-LAYOUT.md` | Source dtype, runtime dtype, memory, and scheduling anchor |
| `docs/sprints/SPRINT-006-REPORT.md` | Context, descriptor, relay, and guard evidence |
| `docs/sprints/SPRINT-006-DEFERRED.md` | Decode and kernel work explicitly left for later |
| `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv` | Source tensor ownership and pack row metadata |
| `gguf-tools/deepseek4-quantize.c` | Existing in-tree FP8/FP4 source dequant reference |
| `tests/test-vectors/official.vec` | Official API selected-token and top-logprob fixtures |
| `ds4.c`, `ds4.h` | Engine guard, CPU reference path, session/test surfaces |
| `ds4_v100_context.*` | Sprint 006 descriptor and execution-policy surface |

## Use Cases

1. **Source dtype proof**: a developer can run a model-less test that proves
   BF16, E8M0, E4M3fn, F8_E4M3_B128, MXFP4, and I32 routing metadata handling.
2. **Guarded oracle open**: a diagnostic test can open the source GGUF through a
   clearly named CPU-only oracle option while normal engine open still rejects
   the model.
3. **Reference dispatch cleanup**: source-layout call sites stop silently using
   legacy `matvec_f16` or `matvec_q8_0` assumptions for BF16/F32/FP8/MXFP4
   tensors.
4. **Official fixture comparison**: the cluster can run one short official
   vector and record first-token equality or a lower-confidence top-K-only
   result.
5. **Future kernel anchor**: later V100 CUDA kernels can compare against a
   canonical source-format oracle rather than re-deriving source semantics.

## Architecture

Sprint 007 adds a sidecar oracle surface and a narrow engine unlock.

```text
source GGUF + pack index
    |
    v
source layout validation
    |
    +--> normal ds4_engine_open
    |        +--> still fails closed
    |
    +--> oracle option + CPU backend only
             |
             v
      ds4_v100_oracle helpers
      - BF16/F32/I32 primitives
      - E8M0 + E4M3fn decode
      - F8_E4M3_B128 blocked rows
      - MXFP4 rows and routed expert slices
             |
             v
      source-aware CPU reference dispatch
             |
             v
      one-slot first-token comparison
      against tests/test-vectors/official.vec
```

### Source-Format Contract

The exact source-format helpers should live in one reusable surface. The sprint
may either promote helpers from `gguf-tools/deepseek4-quantize.c` into
`gguf-tools/quants.[ch]` or introduce a small shared `ds4_source_formats.[ch]`
module if that keeps build linkage cleaner. In either case, the quantizer and
oracle must use the same functions.

Required helper semantics:

| Format | Required Behavior |
|---|---|
| BF16 | bit-exact `uint16_t` to F32 expansion |
| F32 | control-path row copy/matvec without reinterpretation |
| I32 | hash-routing table row reads with bounds checks |
| E8M0 | block-scale decode matching the in-tree quantizer |
| E4M3fn | scalar decode matching the in-tree quantizer |
| F8_E4M3_B128 | 128-value block dequant with scale span validation |
| MXFP4 | 32-value block dequant using the in-tree FP4 table and scale semantics |

Any mismatch with `gguf-tools/deepseek4-quantize.c::dequant_fp8_weight` or
`::dequant_fp4_weight` is a STOP unless the implementation finds and documents
an authoritative source-layout correction.

### Oracle Boundary

The oracle is CPU-only and diagnostic. It may read host-side GGUF or pack bytes
for correctness in this sprint. That is a deliberate shortcut and must be
reported as such; it does not replace Sprint 006's device-resident production
goal.

The engine unlock should be a visibly named option, for example:

```c
bool v100_source_oracle;
const char *v100_source_oracle_unlock_token;
```

The unlock must be accepted only when:

- the model is the recognized V100 source layout;
- backend is CPU;
- MTP is not requested;
- the unlock token matches the code-level diagnostic constant;
- the caller is a test or dedicated oracle tool path.

Oracle-only engines must refuse normal generation, server use, bench/eval use,
and speculative/MTP paths with a clear message.

### Reference Dispatch

The implementation should avoid rewriting the legacy path. Use explicit
source-layout dispatch so legacy q2/q4 and existing Q8/F16 paths keep working.

Minimum source-aware reference coverage:

- token embedding BF16 row gather;
- HC attention/FFN/output control F32 matvecs;
- RMSNorm F32 weights;
- router F32 control matvec and I32 hash-routing metadata;
- attention Q/KV/output F8_E4M3_B128 matvecs;
- compressor/indexer BF16/F32 families as soon as they are reached by the
  single-token oracle;
- shared expert F8_E4M3_B128 matvecs;
- routed expert MXFP4 gate/up/down slices for selected experts;
- output HC F32 control and BF16 output head.

If a source-layout tensor reaches a legacy helper with the wrong dtype, the
oracle path must fail closed rather than reinterpret bytes.

## Implementation

### Phase 0: Baseline And Semantics Lock

**Files:**
- `gguf-tools/deepseek4-quantize.c`
- `gguf-tools/quants.h`
- `gguf-tools/quants.c`
- `tests/source_dtypes_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Extract or share E8M0, E4M3fn, F8_E4M3_B128, and MXFP4 decode helpers.
- [ ] Keep BF16/F16 conversion helpers available from the same shared surface.
- [ ] Add synthetic expected-value tests for BF16, E8M0, E4M3fn, F8 block rows,
      MXFP4 rows, and malformed spans.
- [ ] Add an explicit check that helper outputs match the current quantizer
      dequant behavior for representative rows.
- [ ] Add I32 routing-row bounds tests.

### Phase 1: Oracle Module And Guard

**Files:**
- `ds4.h`
- `ds4.c`
- `ds4_v100_oracle.h`
- `ds4_v100_oracle.c`
- `tests/v100_oracle_guard_smoke.c`
- `Makefile`

**Tasks:**
- [ ] Add the explicit oracle option and unlock token field to
      `ds4_engine_options`.
- [ ] Preserve the existing source-layout guard for all normal engine opens.
- [ ] Permit oracle open only for CPU backend, source layout, matching token, and
      no MTP.
- [ ] Mark oracle engines as oracle-only and make normal generation/session
      paths reject them.
- [ ] Add guard tests for normal rejection, wrong-token rejection, and oracle
      rejection from non-CPU or MTP paths.

### Phase 2: Source-Aware Reference Dispatch

**Files:**
- `ds4.c`
- `ds4_v100_oracle.h`
- `ds4_v100_oracle.c`
- `tests/v100_oracle_dispatch_smoke.c`

**Tasks:**
- [ ] Replace the source-layout embedding path with BF16-aware dispatch while
      leaving legacy behavior unchanged.
- [ ] Add F32 control matvec dispatch for HC control, router control, and output
      HC control.
- [ ] Add F8_E4M3_B128 dense/reference matvec dispatch for attention and shared
      expert families.
- [ ] Add MXFP4 routed expert reference slices for selected expert rows.
- [ ] Add BF16 matvec/row support for source output head and compressor/indexer
      tensors reached by the oracle.
- [ ] Fail closed on unsupported `(source dtype, tensor family)` pairings.

### Phase 3: One-Token Oracle Driver

**Files:**
- `tools/ds4-v100-oracle-decode.c`
- `tests/ds4_test.c`
- `Makefile`

**Tasks:**
- [ ] Add a dedicated diagnostic tool or test mode that opens the source model
      with the oracle unlock and runs one first-token decode.
- [ ] Reuse `tests/test-vectors/official.vec` parsing for selected-token and
      top-logprob evidence.
- [ ] Prefer exact first-token match for `SHIP`; report top-K-only membership as
      `EXTEND`.
- [ ] Emit JSON/text evidence with prompt id, selected token bytes, local top-K,
      official top-K membership, timing, and any full-logit oracle metrics if
      supplied.
- [ ] Cap the cluster oracle run to short fixtures and one decode step by
      default.

### Phase 4: Validation And Report

**Files:**
- `docs/sprints/drafts/SPRINT-007-DTYPE-SMOKE.log`
- `docs/sprints/drafts/SPRINT-007-DISPATCH.log`
- `docs/sprints/drafts/SPRINT-007-ORACLE.log`
- `docs/sprints/drafts/SPRINT-007-GUARD.log`
- `docs/sprints/SPRINT-007-REPORT.md`
- `docs/sprints/VISION.md`

**Tasks:**
- [ ] Run local model-less tests and `git diff --check`.
- [ ] Run focused legacy no-regression tests for the existing supported model
      path if the local model is available; otherwise record the missing model.
- [ ] Build on the V100 cluster with `CUDA_ARCH=sm_70` even though the oracle is
      CPU-only, to prove the tree remains cluster-buildable.
- [ ] Run dtype/dispatch/guard tests on the cluster.
- [ ] Run the one-token source oracle against `/models/DSv4-Flash-256e-fixed.gguf`
      and archive the log.
- [ ] Write `SPRINT-007-REPORT.md` with verdict, evidence, deviations, and the
      Sprint 008 handoff.

## Files Summary

| File | Action | Purpose |
|---|---|---|
| `gguf-tools/quants.h` / `.c` or `ds4_source_formats.h` / `.c` | Modify/Create | Shared BF16/E8M0/E4M3fn/F8/MXFP4 source-format helpers |
| `gguf-tools/deepseek4-quantize.c` | Modify | Reuse shared helpers or document exact parity with them |
| `ds4.h` | Modify | Add explicit oracle option fields |
| `ds4.c` | Modify | Preserve guard, add oracle-only engine handling, source-aware reference dispatch |
| `ds4_v100_oracle.h` / `.c` | Create | Oracle primitives, descriptor checks, and reference matvec helpers |
| `tools/ds4-v100-oracle-decode.c` | Create | Dedicated one-token diagnostic driver |
| `tests/source_dtypes_smoke.c` | Create | Model-less source-format semantics tests |
| `tests/v100_oracle_guard_smoke.c` | Create | Guard and unlock-token tests |
| `tests/v100_oracle_dispatch_smoke.c` | Create | Source dispatch and fail-closed tests |
| `tests/ds4_test.c` | Modify | Optional official-vector oracle integration |
| `Makefile` | Modify | Build new helpers, tests, and tool |
| `docs/sprints/SPRINT-007-REPORT.md` | Create | Sprint verdict and evidence |
| `docs/sprints/VISION.md` | Modify | Record Sprint 007 outcome after execution |

## Definition Of Done

- [ ] Shared source-format helpers define exact BF16, E8M0, E4M3fn,
      F8_E4M3_B128, MXFP4, and I32 routing-row behavior.
- [ ] Helper tests include normal values, zero/negative zero where applicable,
      infinities/NaNs where applicable, bad shape/span failures, and scale-span
      bounds failures.
- [ ] The quantizer and oracle use the same helper semantics or have an explicit
      byte-parity test against the existing quantizer dequant behavior.
- [ ] Source-layout BF16 embeddings and output weights are no longer interpreted
      as F16 or Q8_0 in the oracle path.
- [ ] Source-layout F32 control tensors are no longer interpreted through F16
      control matvecs in the oracle path.
- [ ] F8_E4M3_B128 dense and MXFP4 routed expert tensors have source-aware
      oracle dispatch or precise fail-closed errors.
- [ ] Normal source-layout engine open still fails with the existing guard
      message when oracle unlock is not present.
- [ ] Oracle unlock is CPU-only, source-layout-only, and rejects MTP, server,
      bench/eval/generation, wrong token, and non-oracle callers.
- [ ] Legacy-layout paths remain unchanged and focused no-regression validation
      is run or explicitly recorded as unavailable.
- [ ] No persistent dequantized F16/F32 copy of large source tensors is created.
- [ ] One short official vector reaches exact first-token match on the cluster
      for `SHIP`; top-K-only evidence is recorded as `EXTEND`, not `SHIP`.
- [ ] Validation logs are archived under `docs/sprints/drafts/`.
- [ ] `git diff --check` passes.
- [ ] `SPRINT-007-REPORT.md` and `VISION.md` are updated.

## Risks And Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---:|---:|---|
| FP8/MXFP4 helper semantics do not match the source model | Medium | High | Pin helpers against `deepseek4-quantize.c`; STOP on unexplained mismatch |
| Official vectors are too weak to prove full logits | High | Medium | Treat exact first-token match as required for `SHIP`; document top-K-only limits |
| Oracle bypass leaks into normal runtime | Medium | High | Code-level token, CPU-only checks, oracle-only engine flag, guard regression tests |
| `ds4.c` dispatch cleanup regresses legacy model support | Medium | High | Keep dispatch conditional on `v100_source_layout`; run legacy no-regression check |
| CPU oracle is too slow for cluster validation | Medium | Medium | Limit to short prompt and one token; record timing and `EXTEND` if too slow |
| Host-side oracle does not prove device-resident production behavior | High | Medium | Report as correctness-only; defer device-side oracle/production kernels |
| Hidden `forward_first_token_cpu` assumptions remain legacy-only | Medium | High | Add per-family dispatch tests and fail closed on unsupported source families |
| Scope drifts into KV, prefill, MTP, server, or performance | High | High | Keep non-goals explicit; use `STOP` if implementation requires those paths |

## Security Considerations

- The oracle unlock token is not a secret; it is an audit mechanism so the
  bypass can be found by grep and tested directly.
- The oracle must not be exposed as a normal CLI/server serving mode.
- Model and pack bytes are read-only.
- All row, scale, stride, and byte-span math must validate overflow and bounds
  before reading source bytes.
- The source-layout guard message should remain stable for existing log and test
  checks.

## Dependencies

- Sprint 006 context, descriptor policy, and guard state.
- Existing source-model pack index and persistent cluster model path.
- Existing official vector fixtures.
- V100 cluster access for final evidence.

## Open Questions

1. Should the helper module live in `gguf-tools/quants.[ch]` or a smaller
   runtime-safe `ds4_source_formats.[ch]`?
2. Can the first-token oracle complete within an acceptable cluster time budget,
   or does Sprint 008 need a device-side oracle before official-vector proof?
3. Do we need to capture a local full-logit oracle after Sprint 007 ships so
   later production kernels have a stronger regression target than official
   top-logprob slices?
