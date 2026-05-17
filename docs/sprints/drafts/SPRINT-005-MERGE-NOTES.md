# SPRINT-005 Merge Notes

## Inputs

- Intent: `docs/sprints/drafts/SPRINT-005-INTENT.md`
- Claude draft: `docs/sprints/drafts/SPRINT-005-CLAUDE-DRAFT.md`
- Codex draft: `docs/sprints/drafts/SPRINT-005-CODEX-DRAFT.md`
- Gemini draft: `docs/sprints/drafts/SPRINT-005-GEMINI-DRAFT.md`
- Claude critique: `docs/sprints/drafts/SPRINT-005-CLAUDE-CRITIQUE.md`
- Codex critique: `docs/sprints/drafts/SPRINT-005-CODEX-CRITIQUE.md`
- Gemini critique: `docs/sprints/drafts/SPRINT-005-GEMINI-CRITIQUE.md`

## Draft Strengths

### Claude

- Strongest outcome contract and phase gating.
- Explicit BF16-to-F32 bit-shift conversion logic.
- Good invalid-input matrix and host-stub-first validation order.
- Clear guardrails against decode, scheduler, KV, FP8, MXFP4, and MTP scope
  creep.

### Codex

- Best architecture framing around a reusable BF16 matrix view descriptor.
- Strong fail-closed diagnostics and risk analysis.
- Strongest argument for proving the probe reads arena-resident bytes, not
  GGUF or host staging.
- Useful probe-only tooling idea to avoid full 145 GiB uploads for narrow
  compute validation.

### Gemini

- Clean API sketch and simple phase progression.
- Good emphasis on a residency-smoke integration point.
- Identified that cluster validation should be mandatory for a resident compute
  sprint when cluster access is available.

## Accepted Critiques

- Use a descriptor/view API instead of a long flat parameter list.
- Make F32 host output the only Sprint 005 output contract.
- Require explicit BF16 bit-pattern tests, including BF16-vs-F16 divergence.
- Require 2-byte alignment and integer-overflow checks in tasks and DoD.
- Require direct CUDA synthetic validation on V100 for `SHIP`.
- Keep `ds4.c` dtype-awareness fixes, HC expansion wrappers, F16 output, and
  device-output variants out of Sprint 005 unless discovered to be strictly
  required for the probe.
- Keep the source-model generation guard active.

## Rejected Critiques Or Cuts

- Broad stream-aware API: rejected for Sprint 005 because production stream
  ownership belongs to the Sprint 006 execution context.
- Multi-tensor BF16 probing: rejected for Sprint 005 because `token_embd.weight`
  already proves arena addressing, BF16 conversion, and real source metadata
  without doubling the validation surface.
- F32 control tensor probe: deferred. It is useful if the BF16 probe fails, but
  it is not required for the first implementation pass.
- Model-less default `make test` cleanup: deferred unless it falls out
  trivially. Sprint 005 already has model-less targeted tests.

## Final Scope

Sprint 005 implements a diagnostic BF16 row-gather probe from
`ds4_gpu_arena`, with:

- a reusable BF16 matrix view descriptor;
- host-stub and CUDA implementations;
- exact BF16 conversion tests;
- local synthetic tests;
- a direct CUDA synthetic test;
- a focused residency-smoke probe mode for `token_embd.weight`;
- V100 cluster validation and durable logs.

The final sprint does not enable source-model decode, prefill, KV, MTP,
multi-slot scheduling, FP8/MXFP4 kernels, HC relay, or production multi-GPU
execution context.
