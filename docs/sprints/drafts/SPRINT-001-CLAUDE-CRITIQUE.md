# SPRINT-001 Drafts — CLAUDE Critique

**Reviewer:** Claude Opus 4.7
**Date:** 2026-05-17
**Scope:** Comparative critique of `SPRINT-001-CODEX-DRAFT.md` and
`SPRINT-001-GEMINI-DRAFT.md` against `SPRINT-001-INTENT.md` and the actual
state of the `ds4` repo (`ds4_cuda.cu`, `ds4.h`, `ds4_gpu.h`, `Makefile`).
Both drafts are evaluated on the same axes: strengths, weaknesses, risk
analysis, missing edge cases, and Definition of Done completeness.

---

## 1. CODEX Draft

### 1.1 Strengths

- **Kill-gate framing is consistent throughout.** Every phase has an explicit
  kill gate, and the sprint is up-front that a STOP verdict is an acceptable
  success outcome (intent §Success Criteria, last bullet). This matches the
  intent's "kill-gated feasibility slice" framing exactly.
- **Non-goals are stated early and repeated.** MXFP4/FP8, NCCL, speculative
  decode, server concurrency, and broad kernel import are explicitly out of
  scope. This directly resolves intent Q4 and Q5 and pre-empts scope creep,
  which `SPRINT-025-V100-SPIKE-DIRECTION.md` identified as the most likely
  spike-killer.
- **Phase 0 byte-estimation gate is the right insertion point.** Estimating
  per-device bytes (weights + KV + scratch + cuBLAS) *before* attempting full
  load is a high-value, low-cost check; if budgets are obviously wrong the
  sprint stops cheaply without the operational cost of staging an 81 GiB
  GGUF. The intent's `Verification Strategy` calls for exactly this.
- **Architecture stays close to existing API contracts.** "Keep tensor
  internals opaque" and "active-device control at graph boundaries, not
  per-kernel" preserve the narrow surface called out in `AGENT.md`. The draft
  resists the temptation to thread device IDs through every primitive.
- **HC payload sizing is correct.** `DS4_N_HC * DS4_N_EMBD * sizeof(float) =
  64 KiB` matches `ds4.h` constants and validates that
  `cudaMemcpyPeerAsync` (or host-staged fallback) suffices — no NCCL needed.
- **Risks list is well-formed.** Six risks each with a mitigation; the
  entanglement-of-global-state risk (Risk 1) is the highest-impact one and is
  correctly identified as the gate between Phase 1 and Phase 2.
- **Security Considerations are appropriately scoped.** Doesn't manufacture
  network/auth concerns where none apply. Calls out parser strictness for
  the layer-split override, which is the only meaningful new input surface.

### 1.2 Weaknesses

- **No phase durations or stop-loss budget.** The intent open question 6
  asks for a stop-loss threshold ("one week without coherent q2 output,
  failure to fit q2, …"). The draft answers none of intent Q1–Q6 in the
  body — it just re-lists them under Open Questions. A sprint plan that
  re-asks the intent's open questions has done less planning than it should.
- **Build-target naming is loose.** The draft uses `make cuda CUDA_ARCH=
  sm_70` consistently but never reconciles with the existing
  `cuda-spark`/`cuda-generic` targets in the Makefile. The intent constraint
  "preserve DS4's appliance model" implies single-device DGX Spark
  / generic CUDA builds must keep working; the draft does not say so.
- **No bit-equivalence requirement for single-device.** The architecture
  argues the refactor should "preserve single-device behavior" but the
  Verification steps in Phase 1 only run `cuda_long_context_smoke` on 1/2/8
  GPUs. There is no *bit-identical* logprob check vs the pre-refactor
  binary. The intent's Verification Strategy edge case 4 — "global CUDA
  state must not leak one device's cuBLAS handle… into another device's
  call" — is hard to validate without a deterministic regression.
- **Layer-plan arithmetic is unspecified.** The draft says "contiguous 43-
  layer split across visible GPUs" with reservation on the final device, but
  never publishes the canonical formula or the exact mapping for the 8-GPU
  case. Whether the split is 6,6,6,5,5,5,5,5 (uneven-front) or 5,5,5,5,5,6,
  6,6 (uneven-back) matters because it interacts with the "reserve headroom
  on the final device" rule the draft itself proposes — those two rules can
  contradict each other. (The CLAUDE draft has the inverse bug; see §2.4.)
- **HC cross-device validation does not exercise all pairs.** Phase 3
  describes "allocates buffers on two devices… copies device A -> device B
  -> host". This only covers one ordered pair. On the V100 SXM2 hybrid
  mesh, peer access is not symmetric across all 56 ordered pairs; a one-
  pair test will not surface the cases that force the host-bounce fallback.
- **No quantitative fit budget.** Phase 2 says "credibly below 32 GB per
  device" without a number. "Credible" is not a kill gate — it's a vibe.
  The intent's edge case "model cache pressure must be visible before
  attempting full 81 GB q2 load" implies a *number* (per-device GiB ceiling
  + KV residency + cuBLAS overhead).
- **Decode coherence gate is hand-wavy.** Phase 4: "exact device-local
  invariants plus a non-garbage decode trace". What is the invariant?
  What constitutes garbage? No reference output, no token-match floor.
- **Files Summary is brief but mutates `Makefile` and `README.md` under
  vague preconditions.** "Update only if the sprint clears the go gate"
  is fine for `README.md` but conflates documentation with code: if the
  sprint fails, the new regression target still exists in the Makefile and
  must be carried forward, so the Makefile is *not* an "only on go" file.
- **No outcome contract / closeout template.** The "stop/go verdict" is
  required by intent Success Criteria but the draft does not say what the
  closing artifact looks like (file path, sections, where memory entries
  go, branch tag). The closeout phase exists but is structureless.

### 1.3 Gaps in Risk Analysis

- **No risk for sm_70 intrinsics rejection.** The codebase contains CUDA
  kernels that may reach for `__nv_bfloat16` or FP8 conversions; the
  closest stated risk is "Risk 1 (refactor destabilizes single-device)"
  which is about state ownership, not architecture-version compile failure.
  The intent calls this out in `Success Criteria` ("or build failures are
  isolated to explicit documented sm70 incompatibilities") — the draft does
  not have a matching risk row.
- **No risk for the V100 pod competing with the existing llama.cpp
  workload.** Per intent Context, the homelab `gpu-01` is shared with the
  llama.cpp deepseek path. A multi-GPU q2 load contends with whatever is
  running. Not a code risk, but a sprint-execution risk that can stall
  Phase 4 for a day.
- **No risk for cuBLAS math-mode drift across devices.** If per-device
  cuBLAS handles are created with default math mode and the existing
  `quality_mode` flag is global, half the devices will silently use TF32
  while the others use FP32. Bit-equivalence then fails for reasons
  unrelated to the refactor.
- **No risk for HMM / range-mapping support differing on V100 vs DGX
  Spark.** `ds4_cuda.cu` has `g_model_range_mapping_supported` that falls
  back if `cudaHostRegister` fails. On the V100 driver stack this fallback
  may be hit on some devices and not others; the draft never validates
  that the per-device flag works correctly post-refactor.
- **No risk for stop-loss exhaustion.** The intent asks for a 1-week stop-
  loss. With six phases and no time budgets, the draft cannot tell anyone
  reading it whether it is on track.
- **No risk for "kernels look right but produce garbage" silent drift.**
  Phase 4's kill-gate language allows a STOP "if first sharded decode is
  immediately incoherent" — but the dangerous case is a sharded decode
  that produces *plausibly coherent* output that is silently degraded.
  There is no logprob comparison or reference-vector check to catch it.

### 1.4 Missing Edge Cases

- **`CUDA_VISIBLE_DEVICES` with non-contiguous subset** (e.g., `0,2,4,6`).
  The plan abstraction must use logical indices `[0..n_visible)`, not
  physical CUDA device IDs. Not addressed.
- **`n_visible == 1` but `g_dev[]` indexed up to 8.** The draft says "
  preserve single-device behavior when only one device is visible" but
  does not describe the data structure invariant that protects unused
  slots from being touched.
- **Single-device fall-back at parse time, not runtime.** If
  `DS4_CUDA_LAYER_SPLIT=6,6,6,5,5,5,5,5` is set on a 1-GPU machine, what
  happens? "Fail closed on malformed input… or device IDs outside the
  visible set" covers it, but a layer-count-mismatch isn't quite the same
  as "malformed".
- **Peer access being asymmetric.** `cudaDeviceCanAccessPeer(A,B)` does
  not imply `(B,A)`. The plan's `peer_ok` cache must store directed pairs;
  the draft says "reachability" which is undirected language.
- **Reserve-headroom rule may shift layers off the final device, but the
  embedding device is fixed.** What if the embedding device is the same as
  the final device when `n_visible == 1`? Trivial case, but the planner's
  invariants are not stated.
- **No mention of the prefetch/upload stream lifecycle per device.**
  `ds4_cuda.cu` creates `g_model_prefetch_stream` and
  `g_model_upload_stream` globally; the draft mentions per-device
  "upload stream" but not the prefetch stream, and does not say what
  happens to in-flight prefetches when the active device changes.

### 1.5 Definition of Done Completeness

Stronger than GEMINI's; weaker than it could be. Concretely:

- ✅ sm_70 build status, device inventory, layer plan print, HC cross-device
  copy, per-device state, q2 fit attempt, verdict file — all present.
- ⚠️ **No bit-equivalence DoD for single-device.** This is the single most
  important regression gate for the refactor.
- ⚠️ **No SHA-256 / checksum gate on the staged GGUF.** A reproducible
  fit/load report requires knowing exactly which file was used.
- ⚠️ **No per-device byte ceiling.** "credible q2 fit" is not testable.
- ⚠️ **No requirement that `cuda-spark` / `cuda-generic` builds still
  work.** Single-device users of the existing targets must not break.
- ⚠️ **No commit/tag/report-file requirement.** "Sprint closeout must
  include the commands run…" but where? Which path? Who signs off?
- ⚠️ **No DoD for the smoke test running in CI / regression target.** A
  test that exists but is not wired into `make cuda-regression` is
  effectively absent for the next contributor.

---

## 2. GEMINI Draft

### 2.1 Strengths

- **Concise and scannable.** The whole draft fits in one screen. For a
  feasibility spike that is supposed to be bounded, the brevity is itself a
  feature; it doesn't pretend the sprint is bigger than it is.
- **Architecture summary is clear.** Layer-Sharded Appliance, per-device
  state ownership, HC boundary at ~64 KiB, narrow API extension. The
  three-bullet description in §Architecture captures the design in fewer
  words than any other draft.
- **Risk for managed-memory KV fallback is correctly identified.** Risk 2
  notes that the existing `ds4_cuda.cu` already has partial managed-memory
  support and proposes using it as a KV-cache fallback. That is genuinely
  useful local knowledge.
- **Mitigation for SM70 perf is honest.** "This sprint is for *fit and
  correctness*; performance tuning is deferred to Sprint 002" is exactly
  the right framing per intent and `SPRINT-025-V100-SPIKE-DIRECTION.md`.
- **Files-summary table is concrete.** Lists every file touched with a
  Modify/New/Refactor verb. Easy to read; easy to estimate.

### 2.2 Weaknesses

- **Severely underspecified.** The draft is roughly 1/8 the length of the
  CLAUDE draft and 1/4 the length of CODEX. Phases are three bullets each;
  there is no per-phase verification, no kill gate, no rollback story, no
  byte budget, no parser surface for layer plans, no peer-access matrix,
  no device-id parsing, no validation that `cudaGetDeviceCount` is honored.
- **64K context claim is unjustified.** DoD item 4: "Per-device memory
  report shows <30GB usage per GPU at 64k context." At 64K context on
  DeepSeek V4 Flash with `q2-imatrix`, KV cache alone is much larger than
  the implicit budget — the draft never estimates it. The CLAUDE draft
  more conservatively cites 8K (or 4K fallback). Promising 64K as a DoD
  without arithmetic is the kind of error that turns the sprint into a
  failure for the wrong reason.
- **Layer-count math is hand-waved.** "5-6 layers per GPU" is approximate;
  the exact mapping (which 3 GPUs get 6 layers, which 5 GPUs get 5; or
  whatever choice) is not stated. The intent fixes `DS4_N_LAYER = 43`;
  this should be exact, not approximate.
- **Kill-gate is in the wrong place.** DoD item 6 says "Kill-Gate Check:
  Performance or simplicity advantage over llama.cpp is documented." But
  performance was explicitly deferred (Risk 3). So either the kill gate
  is "documented", which is too soft to be a gate at all, or it conflicts
  with the deferral. Either way, the actual feasibility kill gates —
  sm_70 builds, q2 fits, decode is coherent — are missing.
- **No predecessor/successor or stop-loss.** The draft has no week budget,
  no answer to intent Q6, and no statement of what triggers a STOP versus
  an EXTEND.
- **Doesn't reconcile with existing build targets.** Mentions "Add SM70
  as a primary target" in the Makefile row without acknowledging the
  existing `cuda`/`cuda-spark`/`cuda-generic` targets. The Makefile
  already supports `make cuda CUDA_ARCH=sm_70`; "primary target" is
  unclear and arguably scope creep.
- **VISION.md proposal is out-of-scope drift.** "docs/sprints/VISION.md"
  is listed as **New** in the Files Summary. The intent explicitly says
  "No vision document exists in this repo. Planning from scratch." A
  vision doc is a separate piece of work; lumping it into a feasibility
  sprint that may end in STOP wastes effort if the verdict is STOP.
- **Doesn't address the model-format decision.** The intent's Q1, Q2, Q5
  all bear on whether the sprint targets `q2-imatrix`, the FP4/FP8 model,
  or both. GEMINI implicitly chooses `q2-imatrix` (by naming it in
  Phase 3 and the DoD) but never *says* so or justifies the choice. The
  intent's Q5 (which kernels to import first) is not even acknowledged.
- **No mention of bit-equivalence regression.** The per-device state
  refactor is the highest-risk change in this sprint, and there is no
  validation that single-device behavior is preserved.
- **`tests/cuda_multi_gpu_smoke.c` is named but unspecified.** What does
  it actually check? The draft says "cross-device allocation, HC copy, and
  basic matmul correctness" — the matmul correctness part is new (CODEX
  doesn't require it) and is either non-trivial (real matmul invariants)
  or vacuous (a single `cublasGemm` against pre-computed reference). The
  draft does not say.
- **No outcome contract.** The sprint either passes the DoD or fails;
  there is no SHIP / EXTEND / STOP distinction, no closeout doc, no memory
  entry, no branch tag.

### 2.3 Gaps in Risk Analysis

Three risks total. Compared to CODEX (six) and the intent's edge-case list
(five named edge cases), this is too thin for a kill-gated spike.

- **No risk for the per-device refactor breaking single-device.** This is
  the single highest-likelihood, highest-impact risk in the sprint. CODEX
  catches it (Risk 1); CLAUDE catches it (Risk 2). GEMINI omits it.
- **No risk for sm_70 build failure.** Risk 3 talks about sm_70
  *performance*, not the (more likely, higher-impact) case where some
  intrinsic refuses to compile.
- **No risk for V100 pod resource contention.** Same gap as CODEX, but
  GEMINI is more vulnerable because its DoD assumes 64K context — which
  needs the whole GPU's memory and cannot run alongside another workload.
- **No risk for peer-access asymmetry.** Risk 1 ("P2P may be disabled")
  is binary; the actual V100 SXM2 case is partial (some pairs P2P, some
  not). The mitigation ("host-staged fallback") is correct but the risk
  description undersells how often the fallback will fire.
- **No risk for managed-memory perf cliff.** Risk 2's mitigation
  (managed KV) sounds safe but managed memory paging on V100 is
  catastrophic for hot-path access. If the mitigation has to fire, the
  sprint silently switches from "fit" to "fit but unusably slow", and
  there is no gate that catches this.
- **No risk for cuBLAS handle / math-mode drift across devices.**
- **No risk for missing the stop-loss.**

### 2.4 Missing Edge Cases

Most of the CODEX edge-case list applies here too, with three additions:

- **64K context KV residency.** No estimate of how much VRAM the KV cache
  consumes at 64K context. For DS4 (`N_LAYER=43`, the per-layer compressed
  KV from the architecture), the order of magnitude is several GiB per
  device at 64K — possibly more than the headroom available. The DoD's
  "<30 GiB per GPU at 64 K" is at best an unverified claim.
- **`tests/cuda_multi_gpu_smoke.c` skip behavior on a 1-GPU host.** If
  the CI / regression env happens to be a single-GPU dev box, the test
  must skip-as-pass, not fail. Not addressed.
- **GGUF file integrity / source.** Where does
  `ds4flash-q2-imatrix.gguf` come from? No URL, no SHA-256, no staging
  step. CODEX and CLAUDE both at least name the upstream
  (`huggingface.co/antirez/deepseek-v4-gguf`).

### 2.5 Definition of Done Completeness

Six items. Inadequate for the scope.

- ✅ sm_70 build, multi-GPU smoke pass, q2 load, per-device memory.
- ⚠️ **Numbers are wrong or unjustified.** 64K context, <30 GiB per GPU.
- ⚠️ **"Coherent token output verified against CPU reference"** is a
  great DoD but undefined. What counts as coherent? Token-match floor?
  Logprob distance? Eyeball? The intent says "compare logits or token
  output" and asks for explicit criteria.
- ⚠️ **The kill gate (DoD #6) is not a gate.** "Performance or
  simplicity advantage over llama.cpp is documented" is too vague to
  pass-or-fail on. CODEX correctly defers performance to Sprint 002 and
  uses fit/correctness as the kill gate; GEMINI confuses the two.
- ⚠️ **Missing: bit-equivalence for single-device decode.**
- ⚠️ **Missing: existing `cuda-spark` / `cuda-generic` still build.**
- ⚠️ **Missing: closeout artifact (report file, verdict, tag, memory).**
- ⚠️ **Missing: GGUF checksum.**
- ⚠️ **Missing: any answer to intent Q3 (embed/output placement) and Q4
  (NCCL). These are in Open Questions, but a sprint plan should answer
  them, not pass them through.**

---

## 3. Comparative Summary

| Axis | CODEX | GEMINI |
|---|---|---|
| Length / detail | Heavy, exhaustive | Brief, summary-level |
| Kill-gate discipline | Strong (every phase) | Weak (one wrong-axis gate) |
| Intent open questions answered | 0 of 6 (re-listed) | 0 of 6 (silent on most) |
| Risk count | 6 well-formed | 3 thin |
| Per-device state design | Specified in prose | Mentioned only |
| Bit-equivalence regression | Missing | Missing |
| GGUF identity / checksum | Missing | Missing |
| Byte budget per device | Qualitative | Asserted (64K, <30 GiB) |
| Closeout artifact / verdict | Required, not templated | Absent |
| Build-target reconciliation | Implicit | Implicit / confusing |
| Files Summary | Conservative; doc gated on go | Includes out-of-scope VISION.md |
| Decode-coherence gate | "non-garbage" | "verified against CPU reference" |
| HC pair coverage | One-pair smoke | Unspecified |

### Top 5 Things Either Draft Should Have But Doesn't

1. **A bit-equivalence regression for single-device decode** before merging
   the per-device refactor. The single highest-leverage protection against
   the per-device refactor silently corrupting the existing path.
2. **A concrete per-device byte ceiling** (e.g., ≤ 28 GiB per device at
   8 K ctx FP16 KV) tied directly to the P4 kill gate, with arithmetic.
3. **GGUF SHA-256 + staging step in P0**, so the reported fit/load can be
   reproduced and so the report's per-device byte numbers can be trusted.
4. **All-pairs HC copy validation**, not a single ordered pair. V100 SXM2
   peer access is asymmetric and partial; the smoke test must exercise
   every `(i,j)` ordering.
5. **An outcome contract / closeout template**: explicit report path,
   required sections, branch tag, memory entry, and SHIP / EXTEND / STOP
   trichotomy. Both drafts ask for a "verdict" without saying what file
   it lives in or what its schema is.

### Recommendation

**CODEX is the stronger starting point**: kill-gate discipline, scope
clarity, risk coverage, and architectural fidelity to `AGENT.md` are all
materially better than GEMINI's. Adopt CODEX as the base and merge in
specifically:

- GEMINI's managed-memory-as-KV-fallback observation (Risk 2),
- GEMINI's files-summary table format,
- the five additions above (bit-equivalence, byte ceiling, GGUF SHA-256,
  all-pairs HC, outcome contract).

GEMINI's draft should not be promoted as-is; its DoD numbers (64K context,
<30 GiB) are likely to set the sprint up for a STOP verdict for the wrong
reason, and its kill gate is on the wrong axis (performance, not fit).
Both drafts re-list rather than answer the intent's open questions; the
final plan should commit to model-format target (q2-imatrix), NCCL
omission, and embed/output placement before sprint kickoff, not after.
