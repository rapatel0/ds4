# SPRINT-005 Deferred Items

This file captures work discussed during Sprint 005 planning but excluded from
the final sprint scope.

## HC Expansion Wrapper

**What:** Add a `token_embd.weight`-specific helper that repeats one gathered
embedding row across the DS4 hyper-connection dimension.

**Why deferred:** Sprint 005 proves resident BF16 row gather and conversion.
HC expansion introduces model-specific output shape behavior that belongs with
the production execution context or embedding integration.

**Target sprint:** Sprint 006 or later.

**Prerequisites:** Verified BF16 resident row gather.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, `ds4_gpu_arena_stub.c`, tests.

## Device-Resident Output Variant

**What:** Add a probe or production primitive that writes F32/F16 output to a
`ds4_gpu_tensor` or future execution-context scratch buffer.

**Why deferred:** Sprint 005 uses host F32 output for simple, inspectable
validation. Device-resident output belongs with stream, scratch, and scheduler
ownership in Sprint 006.

**Target sprint:** Sprint 006.

**Prerequisites:** Verified host-output resident BF16 probe.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, future execution-context files.

## Stream-Aware Probe API

**What:** Let the CUDA probe accept an explicit stream and integrate with
future asynchronous scheduling.

**Why deferred:** Sprint 005 intentionally avoids owning stream semantics.
Default-stream diagnostic behavior is enough for correctness proof.

**Target sprint:** Sprint 006.

**Prerequisites:** Multi-GPU execution context design.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`.

## Source-Layout `ds4.c` Embedding Dtype Fix

**What:** Make any existing CPU or graph diagnostic embedding path dispatch on
tensor dtype so source BF16 embedding rows are not interpreted as F16.

**Why deferred:** The source-model generation guard still blocks normal decode,
and Sprint 005 should keep the proof isolated to resident arena bytes. This fix
is important before enabling source decode, but not required for the row-gather
probe.

**Target sprint:** Sprint 006 or Sprint 007.

**Prerequisites:** Resident BF16 probe and decision on how source-layout graph
execution replaces legacy mapped-weight helpers.

**Files:** `ds4.c`, `ds4_cuda.cu`, tests.

## F16 Output Mode

**What:** Add optional BF16-to-F16 output for hidden-context relay or activation
compatibility.

**Why deferred:** Sprint 005 must prove one output contract. Host F32 is easier
to verify and avoids premature activation-format decisions.

**Target sprint:** Future execution-context sprint.

**Prerequisites:** Correct host F32 probe and scheduler output requirements.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, tests.

## F32 Control Tensor Probe

**What:** Add a resident F32 probe for a small control tensor such as
`output_norm.weight` to isolate arena addressing from BF16 conversion.

**Why deferred:** Useful as a diagnostic if BF16 probe failures are ambiguous,
but not required for the first compute proof.

**Target sprint:** Future if needed.

**Prerequisites:** BF16 row-gather implementation or a concrete failure that
requires isolation.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, tools, tests.

## Additional BF16 Tensor Families

**What:** Probe smaller BF16 tensors such as compressor or indexer projection
weights after the embedding path.

**Why deferred:** `token_embd.weight` already proves resident BF16 addressing
and conversion. More tensor families should wait until the execution context
needs them.

**Target sprint:** Sprint 006+.

**Prerequisites:** Verified `token_embd.weight` probe.

**Files:** `tools/ds4-v100-residency-smoke.c`, `ds4_cuda.cu`, tests.

## FP8 And MXFP4 Source Compute Probes

**What:** Add resident compute probes for F8_E4M3_B128 dense tensors and MXFP4
routed expert tensors.

**Why deferred:** These are larger correctness and kernel-scheduling problems.
BF16 is the first low-risk contract proof.

**Target sprint:** Future source-format kernel sprints.

**Prerequisites:** BF16 resident probe and multi-GPU execution context.

**Files:** `ds4_gpu.h`, `ds4_cuda.cu`, future kernel files, tests.

## Model-Less Default `make test`

**What:** Split the default test target so parser/unit/smoke tests can run
without `ds4flash.gguf`.

**Why deferred:** Sprint 005 adds model-less targeted tests, but the broader
default test cleanup is independent and should not distract from resident
compute proof.

**Target sprint:** Test hardening sprint or opportunistic follow-up.

**Prerequisites:** None.

**Files:** `Makefile`, `tests/ds4_test.c`.

## Summary

| Item | Target Sprint | Blocker |
|---|---|---|
| HC expansion wrapper | Sprint 006+ | Needs BF16 row gather proof |
| Device-resident output variant | Sprint 006 | Needs execution-context scratch ownership |
| Stream-aware probe API | Sprint 006 | Needs multi-GPU execution context |
| Source-layout `ds4.c` embedding dtype fix | Sprint 006/007 | Needs source-layout execution integration decision |
| F16 output mode | Future | Needs activation-format requirements |
| F32 control tensor probe | Future if needed | Only needed for diagnostic isolation |
| Additional BF16 tensor families | Sprint 006+ | Needs first BF16 probe |
| FP8 and MXFP4 source compute probes | Future | Needs BF16 proof and kernel plan |
| Model-less default `make test` | Future | Independent test hardening |
