# Structural Extraction Plan — from "smoke is the appliance" to "appliance + library + one-off smokes"

## The diagnosis (the why)

The 23k-line `tools/ds4-v100-tp-ep-full-layer-smoke.cu` is the structural cause
of nearly every problem we keep solving symptomatically:

- **The flag matrix.** 86 gates accumulate because adding a gate to the smoke
  is cheap; moving promoted code into the runtime library was never enforced.
- **Bugs hide in the gate combinations.** A2/A6/A3 investigations all hit the
  same wall: which code actually runs depends on gate intersections you can't
  see from a flag name.
- **Performance is hard to improve.** Tight inner loops are buried under
  gate branches; profiling tells you "the function is slow" but the function
  is 1,000 lines doing eight different things.
- **The "runtime library" is decorative.** `ds4_v100_tp_runtime.{cu,h}` exists
  (~2k lines, 9 kernels), but the smoke duplicates everything (100 kernels,
  0 overlap with the runtime library). The serving appliance launcher exec's
  the smoke, not the library.
- **Experiment ≠ production gets blurred.** The smoke is simultaneously the
  test harness and the production binary. New experiments accumulate in the
  same file that serves traffic.

## The insight that frames the fix

**Smoke tests should be one-off files**, not accumulated paths in a serving
binary. We want **one appliance at the end** — not a codebase targeting 100
different run options.

The corollary: **experiments live in new files; promoted code lives in the
runtime library; the appliance is thin and stable.**

## Target architecture

The directory IS the namespace. No redundant `ds4_v100_` prefix on files —
the whole repo is DS4-V4 inference on V100, and the Metal side already lives
under `metal/` separately. Existing dirs that we slot into: `kernels/` (with
its existing `turbomind/` and `tc-grid/` subdirs untouched) and `tests/`.

```
kernels/                                 # already exists (turbomind, tc-grid live here)
  v100/                                  # NEW — pure kernels, no orchestration
    norm.cuh                             # RMS-norm variants
    hc_mix.cuh                           # HC pre/post/sinkhorn split
    attention.cuh                        # Q/KV projection, attention compute
    ep_compose.cuh                       # routed FFN, dispatch, combine
    router.cuh                           # router select + bias + hash
    mtp.cuh                              # MTP head
    collective.cuh                       # NCCL broadcast/allgather helpers
    fill_pack.cuh                        # fill + pack utilities
    diagnostics.cuh                      # parity audit kernels (optional include)

engine/                                  # NEW — inference logic (sublayer orchestration)
  decode_loop.cu                         # per-step orchestration
  hc_current.cu                          # HC-current sublayer
  attention_step.cu                      # attention sublayer
  ep_step.cu                             # EP compose / MoE sublayer
  router_step.cu                         # router + route plan
  mtp_step.cu                            # MTP head
  collective_plumbing.cu                 # NCCL comm setup / teardown
  context.{cu,h}                         # per-rank state, buffers (absorbs
                                         # the current ds4_v100_context_cuda.cu)
  api.h                                  # narrow public API the appliance calls

appliance/                               # NEW — the production binary
  main.cu                                # THE main entry point. ~300 lines:
                                         #   parse env into options
                                         #   build engine context
                                         #   start HTTP server
                                         #   loop: request → slot → engine_decode_step
  http_server.cu                         # HTTP serving glue
  request_scheduler.cu                   # batch / slot orchestration
  options.h                              # the ~10 real runtime knobs

smokes/                                  # NEW at repo root — one-off experiments
  2026-05-29_sprint_485_a1_attn_norm_rank_local.cu
  2026-06-02_sprint_487_router_logits_fused.cu
  ...                                    # each self-contained, includes from
                                         # kernels/v100/ and (optionally) engine/,
                                         # deleted when sprint closes
```

**The main file is `appliance/main.cu`** — the only file with `int main()`. It
is what `tools/ds4-v100-run-appliance.sh` exec's. Reading it top-to-bottom
tells you what the binary does; following `engine_decode_step()` from
`engine/api.h` takes you into the decode loop in `engine/decode_loop.cu`; from
there into the individual `engine/<sublayer>.cu` files; and the kernels are
at the leaves in `kernels/v100/`. The "where does control flow start" answer
is unambiguous: open `appliance/main.cu`.

Symbol naming (separate from file naming): exported C symbols still want a
short prefix to avoid collisions with system libraries — `ds4_` is plenty
(e.g., `ds4_decode_step()`, `ds4_init_context()`). Static / file-local
functions take no prefix. No `ds4_v100_` on symbols either — same logic.

Existing root files (`ds4.h`, `ds4_v100_tp_runtime.cu`, `ds4_v100_context.h`,
etc.) get absorbed during extraction: the runtime API moves to `engine/api.h`,
the context moves to `engine/context.{cu,h}`, the rest of `ds4_v100_*` either
moves into the appropriate directory or gets deleted as legacy. The `ds4_v100_`
prefix on the existing root files is the same noise we're getting rid of
everywhere else; extraction is the chance to drop it.

Key properties of the target:

- **`kernels/v100/*.cuh` are pure kernels.** No options, no flags, no
  dispatch. Each kernel does one thing and is callable from anywhere
  (appliance, engine, smokes).
- **`engine/*.cu` are inference logic.** They call kernels, manage per-step
  state, and coordinate collectives. No CLI parsing, no HTTP.
- **The appliance is thin.** Target: ≤ **3,000 lines total** across all
  appliance files combined. `main.cu` parses env → builds engine context →
  loops over HTTP requests → calls the engine API. That's it.
- **Smokes are one-off.** New file per experiment, named with date + sprint +
  topic. The smoke `#includes` kernels and (optionally) engine sublayers,
  runs its own A/B harness, writes a status report, then **gets deleted**
  when the sprint closes. The appliance never grows. (Old smokes are
  recoverable from git history; no `legacy/` directory needed.)

## The discipline rules (going-forward, baked into the plan)

1. **No new flags in the appliance.** Real operator knobs (slot count, context
   length, model path, NCCL env) only. Anything that selects between two code
   paths is an experiment, and experiments live in smokes, not the appliance.
2. **Smokes are time-boxed and self-contained.** A smoke is created at the
   start of a sprint, exists only for that sprint's experiment, and is deleted
   or archived when the sprint resolves. No accumulation.
3. **Promotion = moving code into the runtime library, not flipping a flag.**
   When a smoke's experiment succeeds, the promotion commit moves the
   relevant logic into the appropriate `ds4_v100_runtime/*.cu` (and kernels
   into `ds4_v100_kernels/*.cuh`) AND deletes the smoke in the same commit.
4. **Rejection = deleting the smoke.** When a smoke's experiment fails, the
   sprint closes by deleting the smoke. No "we might come back to it." If we
   come back to it, we write a new smoke.
5. **The runtime library has a narrow API.** `ds4_v100_runtime_api.h` is the
   contract. Adding a function to that API is a deliberate decision, not a
   reflex.
6. **Each `ds4_v100_runtime/*.cu` file is bounded.** Soft limit: **2,000
   lines per file**. If a sublayer file grows past that, it's a signal to
   split. Hard limit: **1,000 lines per function**. The current
   `parse_args` (861 lines) and `main` (767 lines) in the smoke are the
   exact thing to never reproduce.

## Phased migration plan

Each phase is **independently shippable** and **bit-exact parity-gated** —
selected-token 256/256 at the reference shape after every commit, plus
`peer_copy_sys_bytes = 0`. Serving must keep working at every step.

### Phase 1 — Kernel extraction (mechanical, low-risk)

Pull all 100 `__global__` definitions out of the smoke into
`ds4_v100_kernels/*.cuh` files, grouped by domain. The smoke `#include`s them
and the code continues to compile and behave identically.

- One commit per kernel group (norm, fill, route, compose, mtp, etc.).
- Per commit: move kernels to header, replace inline defs with includes,
  build, validate parity.
- Each kernel file is bounded to ≤ 1,500 lines.
- Smoke shrinks by ~3,000–4,000 lines through Phase 1.
- Estimated: **2–3 sprints**.

**Out of scope this phase:** changing any kernel's behavior, signature, or
naming. Pure mechanical extraction.

### Phase 2 — Engine sublayer extraction (substantive)

Move the decode loop + sublayer orchestration from the smoke into
`engine/*.cu`. The smoke now calls engine functions instead of
having its own copies of the orchestration.

This is the heaviest phase because the orchestration in the smoke is tangled
with flag dispatch. Approach:

- One sublayer per sprint: HC-current → attention → EP → router → MTP.
- For each sublayer:
  - Define the narrow API in `engine/api.h` for what the appliance needs
    to call.
  - Implement the function in `engine/<sublayer>_step.cu`, pulling the
    orchestration code out of the smoke.
  - **Carry only the promoted code path** — not the gate alternatives. The
    promoted path is what runs in serving today; that's the only behavior
    that needs to be bit-exact-preserved. Rejected/abandoned paths are not
    moved; they stay in the smoke until the smoke is retired in Phase 5.
  - The smoke calls the new `engine/*.cu` function for that sublayer instead
    of running its own copy.
  - Parity-validate.
- After all sublayers extracted: the smoke is reduced to harness +
  parse_args + the (still-existing) gate flags that select alternative
  paths.
- Estimated: **5–6 sprints** (one per sublayer + integration).

**Out of scope this phase:** killing flags. We keep the smoke's flag-driven
alternatives alive so we don't break experiments mid-flight; flag cleanup
happens in Phase 3.

### Phase 3 — Flag cleanup post-extraction (the cleanup sprint, done right)

With the promoted serving paths cleanly in `engine/`, the smoke's remaining
content is mostly dead alternative paths gated by flags that no longer serve
a promoted purpose. Apply the same six-bucket framework from
`docs/sprints/archive/TEMP_CODE_CLEANUP_PROMPT.md` to the now-shrunken smoke:

- Promoted → delete (the engine is the promoted version now).
- Rejected-terminal → delete.
- Dormant-revive → either move to `engine/` or delete (recover from git if
  it ever comes back).
- Diagnostic → move to `kernels/v100/diagnostics.cuh` under a single
  `DS4_DIAGNOSTICS` flag, or delete.
- Runtime knob → move to `appliance/options.h`.
- Experimental-alive → move to a fresh smoke in `smokes/`.
- Estimated: **1–2 sprints.**

### Phase 4 — Build the new appliance binary

`appliance/main.cu` is built and validated alongside the smoke for one sprint.
Once parity is confirmed across the full reference shape, the launcher
(`tools/ds4-v100-run-appliance.sh`) is flipped to exec the appliance instead
of the smoke.

- Appliance target size: ≤ 3,000 lines combined across the appliance
  directory.
- Parity-gated alongside the smoke through this phase — both binaries must
  pass selected-token 256/256 against control.
- Estimated: **1–2 sprints.**

### Phase 5 — Retire the smoke

Once the appliance has been the production serving binary for one sprint with
no regressions, the smoke is deleted. Its remaining non-flag content
(parse_args, harness machinery) either moved into a much smaller
`smokes/full_layer_harness.cu` or absorbed into the per-experiment smokes
that need it.

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` ceases to exist.

- Estimated: **1 sprint.**

## Total estimated effort

**10–14 sprints** of disciplined extraction. Roughly 2 calendar months at the
current sprint cadence if executed without parallel optimization work, or
3–4 months if interleaved with continued perf sprints (recommended — see
"Concurrent perf work" below).

Yes, that's a lot. But the alternative is the next 100 sprints of optimization
work continuing to compound on the same 23k-line file, with each new
investigation re-paying the cost of figuring out which code actually runs.
**This unblocks the perf work — it isn't an alternative to it.**

## Concurrent perf work — how to not freeze the docket

Phase 2 sublayer extraction is the heavy lift, but the extractor can be a
different agent / sprint slot than the perf docket. Suggested concurrency:

- **Extraction sprints** focus on one sublayer at a time, parity-gated. They
  do not change behavior.
- **Perf sprints** continue from `TEMP_POST_SWEEP_DOCKET.md` items (A1
  norms, EP-compose-compact, A4b row-parallel, etc.) — but they target
  whichever surface is currently in the runtime library (and therefore
  cleanly editable). The first sublayer extracted becomes the first
  surface where perf sprints land cleanly.
- **No new flags in the smoke** during extraction. New experiments go into
  `smokes/` as one-off files per the new discipline, even though the engine
  isn't complete yet. This stops the bleeding.

## What this plan does NOT do (out of scope)

- **Does not change kernel logic.** Phase 1–2 are extraction; behavioral
  changes are a different sprint class.
- **Does not redesign the runtime API.** The narrow API in
  `ds4_v100_runtime_api.h` is determined by what the appliance needs to call;
  no API design exercise.
- **Does not introduce C++ or templates.** `AGENT.md` rule. Kernel headers
  are C-compatible `.cuh` with `__global__` definitions and helper
  `__device__` inline functions, but the host orchestration stays C.
- **Does not block A2/A3/sprint 482 / docket #3 work in flight.** Extraction
  doesn't require the docket to pause; both run in parallel.

## Definition of done

The plan succeeds when:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu` no longer exists.
- The appliance binary is ≤ 3,000 lines combined under `appliance/`.
- `engine/` is the inference logic, with each `*_step.cu` ≤ 2,000 lines.
- `kernels/v100/` is pure kernels, each file ≤ 1,500 lines.
- The flag count in the appliance is ≤ 15 (real runtime knobs only).
- New experimental work lands in `smokes/<date>_sprint_<NNN>_<topic>.cu`
  files that are deleted when the sprint closes.
- Promotion commits move code from a smoke into `engine/` (or
  `kernels/v100/`) AND delete the smoke in the same commit.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Extraction sprint introduces a parity regression | Bit-exact 256/256 gate per commit; back out + reclassify on miss |
| Phase 2 takes longer than estimated because orchestration is more tangled than expected | Allow sublayer extractions to ship partial: a sublayer may move 80% of its logic, with the remaining 20% (the messiest gate combinations) left in the smoke for Phase 3 cleanup |
| The "no new flags in the smoke" rule conflicts with a perf sprint mid-extraction | New experiments go to `tests/smokes/` immediately, even before extraction is complete. The discipline starts now, not at the end of the plan. |
| Concurrent perf work touches code being extracted | Extraction sprints announce their sublayer at start; perf sprints either target already-extracted sublayers or wait one sprint |
| The appliance binary's parity vs the smoke takes more than one sprint to confirm | Phase 4 is allowed to take 2 sprints; parity must be ironclad before retiring the smoke |

## One-line summary

Extract kernels (Phase 1), then sublayers (Phase 2), then clean the residue
(Phase 3), then build a thin appliance (Phase 4), then delete the smoke
(Phase 5) — under bit-exact parity throughout, with the discipline that new
experiments are one-off files in `tests/smokes/`, never new flags in the
appliance.
