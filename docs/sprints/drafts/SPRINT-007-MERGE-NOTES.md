# SPRINT-007 Merge Notes

## Inputs

- Intent: `docs/sprints/drafts/SPRINT-007-INTENT.md`
- Drafts:
  - `docs/sprints/drafts/SPRINT-007-CLAUDE-DRAFT.md`
  - `docs/sprints/drafts/SPRINT-007-CODEX-DRAFT.md`
  - `docs/sprints/drafts/SPRINT-007-GEMINI-DRAFT.md`
- Critiques:
  - `docs/sprints/drafts/SPRINT-007-CLAUDE-CRITIQUE.md`
  - `docs/sprints/drafts/SPRINT-007-CODEX-CRITIQUE.md`
  - `docs/sprints/drafts/SPRINT-007-GEMINI-CRITIQUE.md`

## Merge Direction

Use the Claude draft as the base because it has the strongest fail-closed
oracle boundary, explicit `SHIP` / `EXTEND` / `STOP` outcomes, and the clearest
treatment of V100 source precision. Pull in Codex's shared-helper strategy so
FP8/MXFP4 semantics live in one reusable helper surface rather than duplicated
runtime snippets. Keep Gemini's concise framing and explicit Sprint 006 context
link, but do not adopt its public CLI flag as the primary unlock surface.

## Accepted Critiques

- Exact source-format semantics are the first gate. `F8_E4M3_B128` and `MXFP4`
  must be pinned to the existing in-tree `gguf-tools/deepseek4-quantize.c`
  dequant behavior before any oracle decode result is trusted.
- The ordinary source-layout guard in `ds4_engine_open()` must remain the
  default behavior. The only bypass is a code-level, CPU-only oracle option
  intended for tests/diagnostics.
- Exact first-token match against an official short vector is the `SHIP` bar.
  Top-K membership without exact first-token match is only `EXTEND`.
- Legacy-layout no-regression is part of the Definition of Done, not just a
  mitigation.
- I32 hash-routing metadata needs explicit coverage, because layers 0-2 depend
  on `ffn_gate_tid2eid`.
- The official vector fixtures are useful but limited: they expose selected
  tokens and top-logprob slices, not full logits. The report must describe that
  limitation instead of overstating proof strength.
- The oracle may read host-side GGUF/pack bytes in this sprint. That is a
  deliberate correctness shortcut and not a replacement for the pure
  device-resident production runtime.

## Rejected Or Deferred Critiques

- A public `ds4` CLI `--verify-source-correctness` flag is deferred. Sprint 007
  should prefer a dedicated diagnostic tool/test path so the feature is not
  mistaken for supported generation.
- Device-side oracle reads from resident V100 arenas are deferred. Sprint 006
  already proved residency and relay; Sprint 007 is focused on source semantics
  and guarded correctness, not CUDA scheduling.
- Full prefill or multi-step official-vector generation is deferred. The first
  proof target is one prompt and one first token. Additional steps require
  prefill/KV work from the next sprint.
- Full-logit oracle comparison is optional. It can be consumed if available, but
  official API fixtures currently do not provide full logits.

## Interview Notes

No additional user interview was conducted during this merge because the active
goal is to continue the planned sprint sequence, and the preceding discussion
already fixed the key technical stance: V100 does not execute BF16/FP8/FP4
natively, source dtype and runtime dtype must stay separate, and correctness
must stop on material uncertainty rather than weakening the guard.

## Final Plan Changes

- Final sprint title: `Source-Layout Single-Slot Decode Oracle`.
- The plan uses a sidecar `ds4_v100_oracle` module for source-format primitive
  tests and diagnostic decode plumbing, while allowing small `ds4.c` changes for
  engine-option guarding and source-aware reference dispatch.
- `SHIP` requires exact first-token match for at least one short official vector
  on the cluster.
- `EXTEND` covers primitive/helper success plus top-K-only evidence or cluster
  blocked/too-slow evidence.
- `STOP` covers unresolved FP8/MXFP4 semantics, unsafe guard bypass, or scope
  drift into prefill/KV/MTP/server/performance.
