# SPRINT-005 Critique: Claude vs Gemini

This critique is based on the two Sprint 005 drafts as written. It focuses on execution quality, scope control, validation depth, and whether each draft is strong enough to drive implementation without filling in too many gaps later.

## Executive Verdict

- Claude is the stronger base draft. It has a clearer sprint boundary, stronger validation contract, and a Definition of Done that is close to executable.
- Gemini has the cleaner API direction. The descriptor-based probe shape is more maintainable than a flat parameter list, but the draft leaves too many critical behaviors ambiguous.
- Best merge direction: use Claude as the base, then pull in Gemini's descriptor idea only if it can be added without widening the sprint.

## Claude Draft

### Strengths

- The sprint boundary is explicit and disciplined. The overview and outcome contract make it clear that Sprint 005 is a bounded BF16 row-gather proof, not a backdoor start on full execution context work (`CLAUDE:19-49`).
- The API semantics are concrete. The draft defines offsets, sizes, row semantics, and error expectations in enough detail to guide implementation and tests (`CLAUDE:95-141`).
- The validation story is strong. Host-stub first, then CUDA synthetic, then optional real-model integration is the right execution order, and the invalid-input matrix is materially useful (`CLAUDE:215-364`).
- The draft protects the current codebase shape well. It explicitly keeps the legacy `model_map` path and the source-model generation guard out of scope (`CLAUDE:157-161`, `CLAUDE:411-417`).

### Weaknesses

- It is somewhat over-specified. The exact phase breakdown, extra report files, and follow-up docs add process weight that is not essential to proving the compute contract (`CLAUDE:194-211`, `CLAUDE:344-382`).
- The output contract is muddy. The prose says host output is the default, then mentions caller-provided device output "with a flag," but the proposed API exposes no such flag (`CLAUDE:171-188`).
- The public API is parameter-heavy. For a diagnostic sprint that may be acceptable, but it is easier to misuse and harder to extend cleanly than a descriptor/view object (`CLAUDE:99-109`, `CLAUDE:126-136`).

### Scope Risks

- Phase 0 and Phase 5 include non-core work that could dilute the sprint if implementation gets tight: build hygiene, optional test-target cleanup, reports, follow-up docs, and `VISION.md` updates (`CLAUDE:194-211`, `CLAUDE:344-362`).
- Marking cluster validation optional for SHIP is pragmatic, but it weakens confidence in a sprint whose purpose is to prove resident compute on V100 hardware (`CLAUDE:317-342`).
- The HC wrapper is reasonable, but it is the first place the draft starts to pull model-specific behavior into what is otherwise a narrow diagnostic probe (`CLAUDE:124-141`).

### Missing Edge Cases

- The draft does not make 2-byte alignment checks explicit for `weight_offset` and stride-derived addresses, even though BF16 access depends on it.
- Integer-overflow safety is mentioned in Security, but it is not elevated into explicit test tasks or DoD checks for `n_rows * n_cols * 4`, `n_hc * n_embd * 4`, or row-offset multiplication (`CLAUDE:438-453`).
- There is no explicit test for `n_cols == 0`, `row_stride_elements < n_cols`, or `weight_bytes` not being divisible by 2.
- The host-output vs future device-output split is described, but no test or acceptance rule makes that behavior unambiguous.

### Definition Of Done Completeness

- Claude's DoD is strong and is the closest of the two drafts to execution-ready (`CLAUDE:388-419`).
- It should add one explicit DoD item for alignment and overflow validation, not just generic bounds checking.
- It should add one explicit rule for what happens when cluster access is available: either real-model validation is required in that case, or the draft should say synthetic CUDA is the full acceptance bar.

## Gemini Draft

### Strengths

- The draft is easy to read quickly. The high-level goal and sprint intent are clear without a lot of scanning (`GEMINI:13-24`).
- The descriptor-based API is cleaner than Claude's flat signature and is a better long-term shape if the probe contract grows later (`GEMINI:35-58`).
- Integrating a probe mode into the residency smoke flow is operationally sensible and could make cluster validation easier to run repeatedly (`GEMINI:114-133`).

### Weaknesses

- Too many critical behaviors are unspecified or ambiguous. The descriptor is only sketched, invalid token handling says "clamping or failing," and the success criteria mention F32/F16 while the API only exposes `float *out_f32` (`GEMINI:17-24`, `GEMINI:40-57`, `GEMINI:96-99`).
- The sprint lacks decision structure. There is no SHIP / EXTEND / STOP contract and no kill gates between phases.
- The validation plan is too thin. It does not spell out the BF16-vs-FP16 divergence case, an invalid-input matrix, or an explicit proof that CUDA computes from arena device memory rather than copied host bytes (`GEMINI:78-133`, `GEMINI:148-157`).
- The draft leans heavily on modifying the residency smoke tool, which adds operational coupling for what should first succeed as an isolated probe.

### Scope Risks

- The overview expands the target to F32/F16 conversion instead of keeping the sprint on one output contract (`GEMINI:17-24`).
- The descriptor abstraction, stream-bearing API, and smoke-tool integration together risk turning a narrow probe into a small execution-framework refactor.
- The DoD effectively makes cluster success mandatory, but the draft provides no EXTEND path if cluster access is blocked (`GEMINI:148-157`).

### Missing Edge Cases

- No explicit coverage for repeated row IDs, row 0, last row, zero-row gather, null pointers, invalid arena state, or insufficient output buffer.
- BF16 test coverage is incomplete compared with Claude: no explicit negative zero, NaN, Inf, or BF16-vs-FP16 divergence case (`GEMINI:84-99`).
- No explicit alignment or integer-overflow checks around offsets, strides, and output-size calculations.
- No real-model oracle comparison against GGUF bytes, and no explicit no-regression requirement for the legacy path or source-model guard.

### Definition Of Done Completeness

- Gemini's DoD is materially weaker than Claude's (`GEMINI:148-157`).
- It needs explicit invalid-input coverage, a no-regression gate, proof of direct arena-device reads, and a cluster-blocked EXTEND outcome.
- Updating `docs/sprints/VISION.md` is fine as close-out work, but it should not substitute for stronger correctness evidence or durable validation artifacts.

## Recommendation

- Use the Claude draft as the base.
- Borrow Gemini's descriptor/view idea if it can replace the flat probe signature without growing the sprint surface.
- Tighten Claude in two places before finalizing: clarify the host-vs-device output contract, and promote alignment/overflow checks from prose into the actual task list and DoD.
