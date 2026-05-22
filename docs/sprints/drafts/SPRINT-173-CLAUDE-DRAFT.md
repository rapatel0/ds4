# Sprint 173 (Claude Draft) - Reusable Fused Routed-FFN Boundary

Date: 2026-05-22
Status: Draft (planning only; no source edits, no cluster jobs, no commits)

## Overview

Sprints 154-172 exhausted the cheap routed-FFN levers. Wrapper-level CUDA Graph
replay, fixed dispatch bypass (`fixed6`), the down/scatter route-reduce
epilogue, small-route construction, stage-count tuning, host stream-per-expert
scheduling, and slot/layer coalescing all proved either flat or regressive on
the production 16-slot/256K served path. The repeated, evidence-backed
conclusion in `VISION.md` is the same: stop tuning wrapper boundaries and change
a *larger* routed-FFN execution boundary.

This sprint builds the first reusable primitive of that larger boundary: a
**routed-FFN execution contract** plus a **bounded implementation that removes
the route-expanded `a_half` global staging buffer** for the production six-route
decode shape. The contract is the deliverable that matters most. Today's local
one-GPU executor and tomorrow's TP/EP primitive must be the *same* boundary with
a different output mode, so this sprint defines the boundary explicitly (input
layout, route-metadata layout, output mode, expert ownership) and proves it by
eliminating one real global intermediate behind it.

This implements **Option 1 (persistent/fused routed-FFN executor)** in a way that
directly seeds **Option 2 (persistent TP/EP boundary)**: the executor's output
mode and expert-ownership fields are the TP/EP adapter, and the existing
`accumulate_out` argument is the partial-output seed already present in the code.

Concretely, the production path at the served decode shape gathers one token's
hidden vector and writes it to global memory six times (once per routed expert)
as `a_half` before the gate/up GEMM reads it back. That route-expansion is pure
staging traffic with no compute value. Removing it is a real, measurable
liveness win and a clean first cut of the fused boundary.

### Non-Goals

- No full monolithic single-kernel gate/up + SiLU + down + reduce executor. That
  is the multi-sprint endpoint; this sprint builds the boundary it will live
  behind and removes the first global intermediate, keeping the validated
  TurboMind GEMMs as the executor's internal steps.
- No real TP/EP topology, multi-GPU wiring, or scheduler overlay changes. The
  partial-output path is exercised only by a synthetic single-GPU route-split
  smoke.
- No MTP changes, no attention/shared-FFN optimization (a read-only liveness
  report for those is in scope; changing them is not).
- No default promotion unless the served A/B clears the decision gate.
- No new public CLI/server API surface (`AGENT.md`: keep public APIs narrow;
  CLI/server must not learn tensor internals).

## Use Cases

1. **Decode hot path (primary).** 16-slot/256K served appliance, per-step async
   pipeline, one slot per stage call. Each routed-FFN call presents
   `total_routes=6`, `active_experts=6`, `max_routes_per_expert=1`. The executor
   runs in `FULL_OUTPUT` mode and writes the complete `[n_tokens, hidden]` F32
   result, removing the `a_half` route-expansion write.
2. **Correctness/regression reference.** Direct replay on the real model and a
   focused routed-FFN smoke confirm the executor produces bit-comparable output
   to the current production path before any served benchmarking.
3. **TP/EP primitive seed (forward-looking, validated synthetically).** The same
   contract in `PARTIAL_OUTPUT_ACCUMULATE` mode lets a future rank own a subset
   of experts/routes and atomically accumulate its partial `[n_tokens, hidden]`
   contribution into a shared output. This sprint proves the contract composes
   by splitting the six routes into two synthetic groups on one GPU and checking
   that accumulated output equals the full-output run.
4. **Liveness audit continuation.** A verbose, opt-in report names which global
   routed-FFN intermediates were materialized vs eliminated on the selected path,
   so future fusion work can keep auditing the boundary instead of re-deriving
   it.

## Architecture

### Current routed-FFN data flow (production gates)

`cuda_tm_routed_mxfp4_packed_impl` (`ds4_cuda.cu:7197`) with
`GATED_SILU=1`, fused interleaved gate/up, `COMPACT_SCHEDULE=1`,
`DOWN_REDUCE_EPILOGUE=0`, `GRAPH=0`:

```text
selected_i32, weights_f32
   -> cuda_tm_build_routes -> offsets, sorted_pairs, sorted_weights   (:7425)
x_f32 (1 token, hidden=4096)
   -> tm_gather_f32_to_f16_kernel -> a_half  [total_routes x hidden] F16  (:5671,:7516)   <-- EXPANDS 1 row to 6
a_half
   -> cuda_tm_grouped_gated_silu_matmul (gate/up) -> mid_half [total_routes x mid=2048] F16 (:7727)
mid_half
   -> cuda_tm_grouped_matmul (down) -> down_routes [total_routes x hidden] F16  (:7796)
down_routes
   -> tm_scatter_sum_weighted_half_to_f32_kernel -> out_f32 [n_tokens x hidden] F32 (:7886)
```

Global expanded intermediates currently live in the scratch arena
(`ds4_cuda.cu:7376-7389`):

| Buffer | Size (6-route) | Status with production gates | This sprint |
|---|---|---|---|
| `a_half` | `total_routes*hidden*2` = 6*4096*2 = 48 KB | **materialized** (6 copies of 1 token) | **eliminated / un-expanded** |
| `gate_out` | 0 | elided by `use_gated_silu` | unchanged |
| `up_out` | 0 | elided by fused gate/up | unchanged |
| `mid_half` | `total_routes*mid*2` = 24 KB | materialized | unchanged this sprint |
| `down_routes` | `total_routes*hidden*2` = 48 KB | materialized | unchanged this sprint |

`a_half` is the correct first target named in the intent: at the served shape it
is six byte-identical copies of one hidden vector, written to global memory only
to be read straight back by the GEMM. It is staging with zero compute value.

### Why `a_half` and not a full monolithic kernel

The `fixed6` (Sprint 170) and `down_6_m16_reduce` (Sprint 171) probes already
proved that bypassing dispatch or fusing the down epilogue alone does not move
the served path; both were flat-to-regressive. The remaining structural waste at
this shape is the redundant activation expansion. Removing it (a) eliminates a
real global intermediate as the intent requires, (b) reuses the
correctness-validated TurboMind gate/up and down GEMMs as internal steps, and (c)
is numerically low-risk because the un-expanded path is the same math the
existing `use_indexed_a` path (`ds4_cuda.cu:7287,7508`) already runs. A full
fused kernel can come later behind the same contract.

### The reusable boundary contract

A new descriptor makes the routed-FFN boundary explicit on the CUDA side
(`ds4_cuda.cu`; CUDA C++ is already used here, so no `AGENT.md` "no C++"
violation — the surface toward `ds4.c`/host stays C-ABI clean):

```c
typedef enum {
    DS4_TM_ROUTED_OUT_FULL = 0,            // write complete [n_tokens, hidden]
    DS4_TM_ROUTED_OUT_PARTIAL_ACCUMULATE = 1  // atomically add partial into shared out
} ds4_tm_routed_output_mode;

typedef struct {
    // Input layout (un-expanded; no route duplication in global memory)
    const ds4_gpu_tensor *x_f32;       // [n_tokens, hidden] F32, or
    const ds4_gpu_tensor *x_row_ptrs;  // [n_tokens] row pointers
    uint32_t n_tokens, hidden, mid;

    // Route metadata layout (built once, shared by all stages of the boundary)
    const ds4_gpu_tensor *selected_i32;  // [n_tokens, n_routes]
    const ds4_gpu_tensor *weights_f32;   // [n_tokens, n_routes]
    uint32_t n_routes;

    // Expert ownership: which experts THIS executor instance is responsible for.
    // Local one-GPU now: [0, n_total_experts). TP/EP later: a subrange/subset.
    uint32_t n_total_experts;
    uint32_t expert_first, expert_count;  // == full range in this sprint

    // Per-expert MXFP4 weight views in the arena
    const ds4_gpu_turbomind_mxfp4_matrix_view *gate_up; // interleaved
    const ds4_gpu_turbomind_mxfp4_matrix_view *down;

    // Output contract
    ds4_tm_routed_output_mode out_mode;
    ds4_gpu_tensor *out_f32;             // [n_tokens, hidden] F32
} ds4_tm_routed_ffn_desc;
```

Mapping to what already exists, so this is an explicit surface over validated
machinery, not a rewrite:

- **Input layout** — the descriptor mandates un-expanded `x_f32`/`x_row_ptrs`;
  route expansion becomes an internal detail the executor is free to skip.
- **Route metadata** — `selected_i32`/`weights_f32` → the existing
  `cuda_tm_build_routes` products (`offsets`, `sorted_pairs`, `sorted_tokens`,
  `sorted_weights`).
- **Output mode** — `FULL` maps to `accumulate_out=0`;
  `PARTIAL_ACCUMULATE` maps to `accumulate_out=1`, which the impl and the
  scatter/reduce kernels (`tm_scatter_sum_weighted_half_to_f32_kernel`,
  `tm_reduce_sum_weighted_*`, `ds4_cuda.cu:7819-7902`) already honor.
- **Expert ownership** — `expert_first/expert_count` describe a route/expert
  subset. Full range this sprint; the field is the EP adapter. No code may
  assume one GPU owns all experts.

The executor function wraps `cuda_tm_routed_mxfp4_packed_impl` (or, more cleanly,
the impl is refactored to take the descriptor) and adds one new internal
behavior: an **un-expanded activation path** that skips
`tm_gather_f32_to_f16_kernel` and feeds the gate/up GEMM the per-token A buffer
plus `sorted_tokens` indices.

### Two implementation slices

- **Slice A — un-expanded activation (minimum, low risk).** Allocate `a_half`
  at `n_tokens*hidden` (not `total_routes*hidden`), cast each token row once with
  `tm_cast_f32_rows_to_f16_kernel` (`ds4_cuda.cu:5656`), and pass `sorted_tokens`
  as `token_indices` into `cuda_tm_grouped_gated_silu_matmul`. This removes the
  6x expanded write at the served shape and is numerically identical to today's
  path (same `__float2half_rn` cast, same routing). It is essentially productizing
  the `use_indexed_a` path *inside the executor contract and compact schedule*,
  which the current `routed_executor` guard explicitly forbids (`!use_indexed_a`,
  `ds4_cuda.cu:7302`).
- **Slice B — in-kernel F32->F16 A-tile load (stretch).** A TurboMind probe
  variant of `ggml_turbomind_ds4_mxfp4_gated_silu_6` that takes the **F32**
  activation pointer and casts to F16 in registers/shared memory as it loads A
  tiles, removing `a_half` entirely (zero global activation staging). Numerically
  identical if it uses round-to-nearest. Implement only if Slice A lands with
  build + correctness budget to spare.

If neither the served gate clears nor a global intermediate can actually be
removed, the executor still ships as the **opt-in TP/EP primitive seed** (the
contract), and the project pivots to persistent TP/EP planning per the gate.

## Implementation

Phased so each phase is independently verifiable and the sprint can stop at a
defensible boundary if the V100 budget runs out.

### Phase 1 - Executor contract / descriptor (host + CUDA, `ds4_cuda.cu`)

1. Add `ds4_tm_routed_output_mode`, `ds4_tm_routed_ffn_desc`, and a dispatch
   function `cuda_tm_routed_ffn_execute(const ds4_tm_routed_ffn_desc *)`.
2. Refactor `cuda_tm_routed_mxfp4_packed_impl` to be callable from the dispatch
   function with no behavior change when `out_mode=FULL`, `expert_first=0`,
   `expert_count=n_total_experts`, and the un-expanded path is off. Keep the old
   signature as a thin shim so callers in `ds4_cuda.cu:8261-8642` are unchanged.
3. Add a new executor mode value to the existing selector
   (`cuda_tm_routed_executor_mode`, `ds4_cuda.cu:5994-6040`):
   `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6` ->
   `DS4_CUDA_TM_ROUTED_EXECUTOR_FUSED6`. Default stays `off`.
4. **Update the launcher allowlist** in `tools/ds4-v100-run-appliance.sh` to
   accept `fused6` (Sprint 170 lesson: the A/B will not start otherwise; this is
   a DoD item, not optional).

### Phase 2 - Slice A: remove route-expanded `a_half`

5. In the new executor mode, for the candidate shape only, allocate `a_half` at
   `n_tokens*hidden` and take the un-expanded branch:
   `tm_cast_f32_rows_to_f16_kernel` + `sorted_tokens` into the gate/up GEMM,
   skipping `tm_gather_f32_to_f16_kernel`.
6. Reuse the existing candidate-shape guard from Sprint 170 (`ds4_cuda.cu:7303`):
   `total_routes==6`, `num_experts==6`, no token-index indirection required by
   the caller, fused interleaved gate/up with gated-SiLU active, `K==hidden==4096`,
   `fused_n==4096`. Outside the guard, fall back to the existing expanded path.
7. Preserve the compact schedule, profiling hooks, and the down + scatter/reduce
   tail unchanged.

### Phase 3 - Liveness instrumentation

8. Under the existing `DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE`
   (`ds4_cuda.cu:6052`), emit a one-shot-per-shape report listing each routed-FFN
   global intermediate and whether it was `materialized` or `eliminated` on the
   selected path, e.g.:
   `ds4: routed-FFN liveness shape=6 a_half=eliminated mid_half=materialized down_routes=materialized scatter=weighted`.
9. (Read-only audit, optional) Add a single log line reporting the F16/F32
   staging footprint of the shared-FFN and attention paths so the dtype/liveness
   audit (Open Question 1) can continue without optimizing those paths now.

### Phase 4 - Partial-output API + synthetic TP/EP smoke

10. Implement `PARTIAL_OUTPUT_ACCUMULATE` by routing `out_mode` to the existing
    `accumulate_out=1` behavior and honoring `expert_first/expert_count` as a
    route/expert subset filter when building the compact schedule.
11. Add a synthetic single-GPU correctness smoke (in `tools/ds4-v100-replay.c` or
    a focused smoke): run the six routes as one `FULL` pass, then as two
    `PARTIAL_ACCUMULATE` passes over disjoint expert subsets into a zeroed output,
    and assert the two results match within the established TurboMind repeat-run
    drift envelope (`~0.016`-`0.022` abs, per Sprint 167). This proves the
    contract composes for EP without any topology change.

### Phase 5 - Stretch: Slice B in-kernel F32->F16 (only if budget remains)

12. Add a probe variant taking the F32 activation pointer; cast in registers
    during A-tile load; declare/export it in the TurboMind ABI
    (`api.cc`, `ggml-turbomind-ds4-probe.cu`, `include/ggml-turbomind-api.h`)
    only if Slice A is already validated.

### Phase 6 - Validation (V100 pod; see Definition of Done)

Build, symbol/selector evidence, replay smoke, focused correctness, then
16-slot/256K served A/B with prompt/generated/continuation split.

## Files Summary

| File | Change | Phase |
|---|---|---|
| `ds4_cuda.cu` | Descriptor + `cuda_tm_routed_ffn_execute`; refactor impl to descriptor; `fused6` mode; un-expanded `a_half` path; partial-output via `accumulate_out` + expert subset; liveness report | 1-4 |
| `tools/ds4-v100-run-appliance.sh` | Launcher allowlist: accept `fused6` (required for A/B to start) | 1 |
| `tools/ds4-v100-replay.c` | Focused routed-FFN executor smoke + synthetic FULL-vs-PARTIAL split correctness check | 4 |
| `kernels/turbomind/ggml-turbomind/ggml-turbomind-ds4-probe.cu` | (Stretch) in-kernel F32->F16 A-load probe kernel | 5 |
| `kernels/turbomind/ggml-turbomind/api.cc` | (Stretch) export new probe ABI entry | 5 |
| `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h` | (Stretch) declare new probe ABI entry | 5 |
| `docs/sprints/SPRINT-173.md` | Sprint record + outcome + decision | 6 |
| `docs/sprints/VISION.md` | Append Sprint 173 result row + revision bump | 6 |
| `docs/sprints/SPRINT-173-FOLLOWUPS.md` | Carry-forward: full monolithic kernel; remove `mid_half`/`down_routes`; real TP/EP wiring | 6 |
| `docs/operations/DS4-V100-APPLIANCE.md` | (Nice-to-have, Sprint 170 follow-up) document `ROUTED_EXECUTOR` values incl. `fused6` | 6 if touched |

Only `ds4_cuda.cu` and the launcher allowlist are required for the minimum
acceptable outcome. The TurboMind ABI files are touched only for Slice B; if
Slice B is skipped, no `.so` rebuild is required and the `tools/ds4-v100-replay`
sm_70 rebuild plus a server restart cover the change.

## Definition of Done

- [ ] `ds4_tm_routed_ffn_desc` contract exists with explicit input layout, route
      metadata, output mode (`FULL`/`PARTIAL_ACCUMULATE`), and expert-ownership
      fields; no code path assumes one GPU owns all experts.
- [ ] `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6` selects the new executor;
      default remains `off`.
- [ ] Launcher allowlist accepts `fused6`.
- [ ] At least one global expanded routed-FFN intermediate is removed/bypassed on
      the candidate path (`a_half` un-expanded for the 6-route shape), with
      liveness-log evidence.
- [ ] `git diff --check` clean; affected host/CUDA files compile where locally
      possible.
- [ ] `tools/ds4-v100-replay` builds on the V100 pod at `CUDA_ARCH=sm_70`
      (and `libggml-turbomind.so` rebuilds + exports the new symbol **iff** Slice
      B/ABI changed).
- [ ] `fused6` selection is visible in a served or replay log.
- [ ] Focused routed-FFN correctness passes on direct replay before any served
      benchmarking; synthetic FULL-vs-PARTIAL split matches within the drift
      envelope.
- [ ] 16-slot/256K served A/B via `tools/ds4-v100-appliance-soak.sh` records
      **prompt, generated, and continuation/decode** tok/s separately, same-binary
      control vs `fused6`, with token match.
- [ ] Result recorded in `docs/sprints/VISION.md` and cluster logs; changes
      committed (by the operator, per task constraints).

### Decision Gate

- **Promote** `fused6` toward default only if continuation/decode tok/s improves
  by **>= 10%** with correctness intact.
- **Keep opt-in as the TP/EP primitive seed** if the executor is correct and a
  real global intermediate is removed but the gain is `< 10%`. The contract is
  still the deliverable.
- **Stop and pivot to persistent TP/EP planning** if the primitive cannot
  actually remove a real global intermediate. (This hard-stop is independent of
  the perf number: a boundary that stages the same memory is not the boundary we
  need.)

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Routed FFN is numerically sensitive (routing + MXFP4 + gate/up/down + activation + route weight + residual). | High | Slice A is bit-comparable by construction (same cast/rounding, same GEMMs, same scatter). Gate served A/B behind direct-replay correctness; require token match. |
| Scope overrun: full fused kernel is multi-sprint. | High | Minimum acceptable = contract + Slice A (un-expanded `a_half`) only. Slice B and monolithic fusion are explicitly stretch/deferred. |
| Re-introducing the `indexed_a` overhead that the executor guard avoids. | Medium | At the 6-route shape the cast is one row; measure with the liveness log and the served A/B. If `indexed_a`-style indexing regresses, fall back to expanded path inside the guard and record it. |
| Architecture repeats the failed one-layer TP overlay. | High | No scheduler overlay, no per-layer copy/sync bolt-on. Partial-output is only the existing `accumulate_out` + expert subset, validated synthetically; no real multi-GPU wiring this sprint. |
| Launcher allowlist omission blocks the A/B (Sprint 170 failure mode). | Medium | Allowlist update is an explicit Phase 1 / DoD item. |
| `a_half` removal saves too little at 6 routes to move tok/s. | Medium | Expected and acceptable: the gate distinguishes "promote" from "keep as TP/EP seed". The contract value does not depend on the 6-route tok/s delta. |
| ABI/`.so` rebuild risk if Slice B is attempted late. | Medium | Slice B is last; if attempted, rebuild `.so` and re-run symbol check + full replay before any served A/B. Skip cleanly if budget is short. |

## Security

- **Local appliance, no new external surface.** The change is device-resident
  inference; no network, auth, or input-trust boundary is added. The new mode is
  opt-in and default-off.
- **No widening of the public API.** Per `AGENT.md`, the descriptor lives on the
  CUDA side; the host/`ds4.c` boundary stays C-ABI clean and CLI/server learn no
  tensor internals. The `fused6` flag is a validation gate, not a permanent
  semantic fork — the executor is intended to converge to the one release path,
  consistent with "do not add permanent semantic variants behind flags."
- **Memory safety.** The un-expanded path must bound A-row indexing by
  `n_tokens` and column access by `hidden`; the in-kernel cast (Slice B) must
  bound tile loads to `[hidden]`. Partial-output accumulation uses atomics into
  F32 output and must zero the buffer exactly once (the impl already guards this
  via `accumulate_out`, `ds4_cuda.cu:7770-7884`).
- **Model quality preserved.** No precision downgrade beyond the validated
  FP16-tensor-core / FP32-accumulate path; V100 has no native BF16/FP8/FP4 tensor
  cores, so compact storage stays compact and expansion happens only in
  registers/shared memory near tensor-core work. Correctness is gated before
  throughput, per the North Star.

## Dependencies

- **Hardware/runtime:** 8x V100-SXM2-32GB pod (`llm/llamacpp-build-8gpu`,
  gpu-01). Builds and served A/B run on the pod, not the laptop. The pod
  currently has no active GPU compute jobs.
- **Toolchain:** `CUDA_ARCH=sm_70` for `tools/ds4-v100-replay`; the TurboMind
  build toolchain only if Slice B changes the ABI.
- **Code preconditions (all met):** `GATED_SILU`, fused interleaved gate/up,
  `COMPACT_SCHEDULE`, the `ROUTED_EXECUTOR` selector + `fixed6` candidate-shape
  guard, `accumulate_out`, `use_indexed_a` cast path, and
  `cuda_tm_build_routes` (`sorted_tokens`, `sorted_pairs`, `sorted_weights`) all
  exist and are validated by Sprints 158-172.
- **Harnesses:** `tools/ds4-v100-appliance-soak.sh` (served A/B),
  `tools/ds4-v100-replay.c` (direct replay + focused smoke),
  `tools/ds4-v100-run-appliance.sh` (launcher + allowlist).
- **Prior follow-up:** `docs/sprints/SPRINT-170-FOLLOWUPS.md` item 1 (persistent/
  fused six-route executor) — prerequisites closed by Sprints 171-172. Item 2
  (runbook flag doc) is the optional nice-to-have above.
- **No dependency** on TP/EP topology, MTP, or any unmerged work.

## Open Questions

1. **Routed-FFN only, or include a shared-FFN/attention liveness table?**
   Recommendation: optimize routed FFN only; add a *read-only* liveness log line
   for shared FFN and attention (Phase 3.9) so the audit continues without
   widening scope. Confirms routed FFN remains the best first fused island
   without committing to changing the others.
2. **Remove `a_half` first, or start with a full fused gate/up+act+down kernel?**
   Recommendation: remove `a_half` first (Slice A). It removes a real global
   intermediate at low numerical risk by reusing validated GEMMs, and it is the
   intent's stated preferred slice. Full fusion is the deferred endpoint behind
   the same contract.
3. **What served A/B threshold terminates Option 1 and forces the TP/EP pivot?**
   Recommendation: keep `>= 10%` continuation/decode for promotion, but make the
   *hard* terminator independent of perf — if the boundary cannot remove a real
   global intermediate, pivot regardless. A boundary that re-stages the same
   memory is not progress toward the architecture.
4. **Partial-output: API shape only, or tested this sprint?**
   Recommendation: implement the API shape **and** validate it with a synthetic
   single-GPU two-way route/expert split (Phase 4). It is cheap, needs no
   topology, and de-risks the TP/EP claim that the local executor is genuinely
   the future primitive — directly addressing the architecture-risk lesson from
   the failed one-layer TP overlay (Sprints 164-165).
