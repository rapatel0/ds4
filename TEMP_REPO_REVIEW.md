# Repo Review — what each thing is, and the methodology in use

## TL;DR

This repo started as `ds4.c`, a hand-written single-file DeepSeek-V4-Flash
inference engine for Metal (Mac). The active work is **a V100 TP/EP serving
appliance** that lives in CUDA files at the root + a giant test/serving binary
in `tools/`. Everything else is one of: vendored kernel libraries, sprint
docs, Python A/B tooling, the paused vLLM reference port, or accumulated
temporary status reports from the sprint loop.

## Root layout, by purpose

### Project guidance — read these first

| File | What it is |
|---|---|
| `README.md` | Project overview |
| `AGENT.md` | **The rules an agent follows when editing this repo.** Includes the "no permanent flag variants," "no C++," "correctness before speed" doctrine. |
| `CONTRIBUTING.md` | How to test changes — correctness and speed tracks |
| `MODEL_CARD.md` | DS4-V4-Flash model details |
| `gpu-profiling-guidance.md` | How to profile GPU work in this project |

### Production-path C/CUDA at the root (the V100 serving runtime)

These are *not* the giant smoke; they are the library the smoke is built on.
A clean, narrow API.

| File | What it is |
|---|---|
| `ds4.h` | Public C API for the inference engine |
| `ds4_cuda.cu` | CUDA-side hooks called by the engine |
| `ds4_gpu.h`, `ds4_pack.h`, `ds4_source_formats.h`, `ds4_turbomind_pack.h` | GPU buffer / weight-packing interfaces |
| `ds4_v100_context.h`, `ds4_v100_context_cuda.cu` | V100-specific context (per-rank state, buffers) |
| `ds4_v100_layer_execute.h`, `ds4_v100_layer_state.h` | Layer execution + state |
| `ds4_v100_mtp.h` | Multi-token-prediction head |
| `ds4_v100_replay.h` | CUDA-graph replay machinery |
| `ds4_v100_scheduler.h` | Request scheduler |
| **`ds4_v100_tp_runtime.cu`, `ds4_v100_tp_runtime.h`** | **The TP/EP runtime called by serving.** This is the production library. |
| `linenoise.h`, `rax.h`, `rax_malloc.h` | Vendored small libraries (line editor, radix tree) |

### `tools/` — the experimental + serving + benchmark surface (96 files)

Three broad classes:

**1. The giant serving + test binary.** `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
is **23,167 lines**. This is where most of the recent sprint work lands —
flags, gates, alternative paths, A/B comparisons — and what the recent code
cleanup discussion is about. It's both a "smoke" test runner *and* the actual
serving binary for the appliance; the same file gates between "exercise this
case" and "serve over HTTP."

**2. Smaller .cu smoke / proxy / workbench binaries.** Each one isolates a
narrower question:

| Pattern | Examples | Purpose |
|---|---|---|
| `*-layer-smoke.cu` | `tp4-layer-smoke`, `tp8-layer-smoke`, `tp8-real-layer-smoke` | per-layer execution isolation |
| `*-collective-smoke.cu` | `tp4-collective-smoke`, `tp8-collective-smoke` | NCCL collective correctness |
| `*-layer-proxy.cu` | `tp4-layer-proxy`, `tp8-layer-proxy` | drop-in proxy used to A/B a path against a baseline |
| `*-workbench.cu` | `tp8-collective-workbench` | "scratchpad" for hand-running sub-experiments |
| `*-turbomind-*.cu` | `tp4-turbomind-layer-smoke`, `tp8-turbomind-ffn-smoke` | exercise the TurboMind SM70 WMMA kernels |
| `*-mtp-*.{c,cu}` | `mtp-attn-smoke`, `mtp-ffn-smoke`, `mtp-forward-smoke`, `mtp-logits-smoke`, … | multi-token-prediction head isolation tests |
| `ds4-v100-tp-runtime-smoke.cu` | runtime API smoke test |
| `appliance-pack.cu`, `turbomind-pack.cu` | weight-packing utilities |

Naming is consistent: **`ds4-v100-<scope>-<topology>-<what>.{cu,c}`**. `tp4`
and `tp8` are tensor-parallel topology widths.

**3. Python A/B + benchmarking tooling.** All `tools/ds4-v100-*.py`:

| Script | Purpose |
|---|---|
| `ds4-v100-http-readiness-check.py` | wait until the serving HTTP comes up |
| `ds4-v100-http-response-parity.py` | strict bit-exact response comparison |
| `ds4-v100-http-response-tolerance.py` | the new relaxed-gate response checker (sprint 478+) |
| `ds4-v100-tp-ep-profile.py` | profile a run, produce the domain timing table |
| `ds4-v100-tp-ep-steady-profile.py` | the de-confounded steady-state profile (sprint ~470s) |
| `ds4-v100-tp-ep-nccl-http-ab.py` | NCCL-gated A/B harness against HTTP |
| `ds4-v100-tp-ep-http-ab.py`, `ds4-v100-tp-ep-true-attn-http-ab.py` | other A/B harnesses |
| `ds4-v100-tp-ep-correctness-gate.py` | sprint 478's gate harness |
| `ds4-v100-tp-ep-reference-parity.py` | compare against the authoritative reference |
| `ds4-v100-tp-ep-active-slot-matrix.py`, `ds4-v100-tp-ep-nccl-kv-matrix.py` | matrix sweeps over slot counts / KV options |
| `ds4-v100-tp-ep-vram-ledger.py` | track VRAM headroom across the run |
| `ds4-v100-tp-experts-ab.py` | TP-experts vs EP A/B test |

**4. Shell harnesses (`tools/*.sh`).** Top-level launchers that put it all
together for a specific test scenario. `ds4-v100-run-appliance.sh` is the
serving entry point (2,338 lines — it's the big "run everything" script).
Others: `*-soak.sh`, `*-smoke.sh`, `*-bench.sh`, the per-shape gates
(`256k-32slot-gate.sh`, `256k-warmed-production-gate.sh`).

### `deploy/` — how it actually runs

| File | What |
|---|---|
| `deploy/v100/ds4-v100-appliance.env.example` | env-var configuration template |
| `deploy/v100/ds4-v100-appliance.k8s.yaml` | Kubernetes pod spec for the V100 pod |
| `deploy/v100/ds4-v100-appliance.service` | systemd service unit |
| `deploy/v100/ds4-v100-build-localpool.pod.yaml` | build-pod manifest |

### `kernels/` — vendored low-level kernel libraries (~450 files)

| Subdir | What |
|---|---|
| `kernels/turbomind/` | LMDeploy's TurboMind SM70 WMMA kernels (the FP16/FP8/quant GEMM kernels you can't easily rewrite). Includes a `ggml-turbomind/` integration layer with `test_tp_split_*.cpp` etc. |
| `kernels/tc-grid/` | Tensor-core grid kernels |

Most of this isn't touched per sprint — it's the underlying compute we sit
on top of. The cleanup discussion didn't include this.

### `docs/` — actual documentation

| Path | What |
|---|---|
| `docs/architecture/` | Layout, communication patterns, bottlenecks — `DS4-V100-LAYOUT.md`, `DS4-V100-TP-EP-BOTTLENECKS.md`, `DS4-V100-TP-EP-LAYER2-COMMUNICATION.md`, `DS4-V100-TP8-INVESTIGATION.md` |
| `docs/operations/` | Operating the appliance — `DS4-V100-APPLIANCE.md` |
| `docs/sprints/VISION.md` | **The north star: where we're going.** |
| `docs/sprints/STATUS.md`, `EXPERIMENT-STATUS.md` | Rolling status |
| `docs/sprints/SPRINT-001.md` … `SPRINT-481.md` | One formal sprint doc per sprint. Plus per-sprint `*-REPORT.md`, `*-SEED.md`, `*-FOLLOWUPS.md`, `*-DEFERRED.md` |
| `docs/sprints/archive/`, `docs/sprints/drafts/` | older / in-progress sprint docs |

### `research/` — the vLLM reference port (Spike A, paused)

`research/1Cat-vLLM/` is the V100-optimized vLLM fork referenced by
`SPIKE_B_STEERING.md` and `ds4-vllm-port-workspace`. Paused, separate program.

### `tests/`, `gguf-tools/`, `metal/`, `speed-bench/`, `misc/`, `third_party/`, `dir-steering/`

Smaller utility / experiment directories. `metal/` is Metal kernel sources
from the original `ds4.c` Metal-on-Mac codebase. `gguf-tools/` is for the
GGUF model format. `speed-bench/` is for benchmarking.

### Build / logs

| Path | What |
|---|---|
| `build/` | build output, currently empty |
| `logs/` | **19,899 log files** from past runs. Local-only artifacts. |

### `SPIKE_B_*.md` — the current optimization program

- `SPIKE_B_STEERING.md` — the canonical optimization steering doc (the one
  I helped construct; defines A1–A6 / B1–B5 / C1–C4 buckets).
- `SPIKE_B_PLAN_ASSESSMENT.md` — running assessment of where we are against
  the plan.

### `TEMP_*.md` — the temporary working set (204 files)

This is the noise you've been seeing. Two sub-categories:

**`TEMP_STATUS_REPORT_NNN.md` (189 of them, numbered 001–479).** One per
sprint, written at the repo root by the executing agent. Each summarizes
that sprint's result. They never get deleted; they accumulate.

**`TEMP_<topic>.md` (~15 of them).** Working docs we write between us:
- **`TEMP_SYS_TRANSPORT_SWEEP.md`** — sprint 479 spec
- **`TEMP_HC_ALLREDUCE_PROMPT.md`, `TEMP_HC_ALLREDUCE_STEER.md`** — the
  rank-local A2/A3 program
- **`TEMP_PARITY_POLICY.md`** — the relaxed parity-gate policy
- **`TEMP_PATTERN_A_PROMOTION_PROMPT.md`** — A2/A3 promotion sprint
- **`TEMP_POST_SWEEP_DOCKET.md`** — the prioritized post-sweep work list
- **`TEMP_A6_RANK_LOCAL_NORM_PROMPT.md`** — A6 specifically
- **`TEMP_CODE_CLEANUP_PROMPT.md`** — the cleanup spec
- **`TEMP_NCCL_BROADCAST_REDUCTION_AUDIT.md`** — the audit from sprint 479
- **`TEMP_GRAPH_PRIOR_INSIGHTS.md`, `TEMP_SPIKE_A_VLLM_PORT.md`,
  `TEMP_SPIKE_B_C_CAPTURE.md`** — older planning artifacts

The cleanup sprint should also address these (or at least define a retention
policy).

---

## Coding methodology — the pattern, as inferred

This is what's actually being practiced, derived from `AGENT.md` plus the
repo's behavior:

### The doctrine (from `AGENT.md`)

1. **Pure C, no C++.** The runtime is hand-written C. Reluctant to add
   abstractions.
2. **Correctness before speed.** "Do not keep a faster path with unexplained
   attention, KV cache, or logits drift." This is why the parity-gate
   discipline matters.
3. **Narrow public APIs.** CLI/server code should not know tensor internals.
4. **No permanent semantic variants behind flags.** **This is the rule that's
   been slipping** — see `TEMP_CODE_CLEANUP_PROMPT.md`. Diagnostic switches
   are explicitly allowed, but "semantic variant" flags should be transient.
5. **Comments instructive and compact, beside the implementation** — not in
   separate design docs. Concretely: load-time invariants, shape constraints,
   collective ordering rules go in `// like-this` comments.

### The sprint loop (in practice)

1. **Plan in `docs/sprints/SPRINT-NNN.md`** (the formal doc).
2. **Implement** a new path in the giant smoke binary, behind a
   default-off `--xxx-gate` flag.
3. **Build** on the V100 pod via the build-pod manifest.
4. **A/B run** using a Python harness (`ds4-v100-tp-ep-*-ab.py` style)
   against the reference shape: 32 slots / 256K / 256 req / 64 tok.
5. **Gate**: previously strict bit-exact selected-token (256/256); since
   sprint 478 the policy is the relaxed agreement gate
   (`TEMP_PARITY_POLICY.md`).
6. **Write `TEMP_STATUS_REPORT_NNN.md`** at the repo root summarizing what
   happened.
7. **Decide**: promote (flip default), reject (leave default-off), or
   experimental (keep evaluating). **The rule that should hold but doesn't:**
   promoted/rejected commits should also remove the flag and the dead
   branch. They haven't, which is why we have the cleanup sprint.
8. **Update** `VISION.md` / `STATUS.md` if the program direction moves.

### Naming conventions

| Convention | Meaning |
|---|---|
| `ds4-v100-` prefix | targeted at the V100 appliance |
| `tp4` / `tp8` | tensor-parallel topology width (4 or 8 GPUs) |
| `-smoke.cu` | "smoke test" — small isolated reproduction binary |
| `-proxy.cu` | drop-in proxy for A/B comparison |
| `-workbench.cu` | scratchpad for hand-experimentation |
| `--xxx-gate` flag | per-sprint default-off feature flag (these accumulate) |
| `*-parity-gate` flag | audit-only flag emitting a comparison log |
| `*-rank-major-*` | uses the per-rank shard layout |
| `*-rank-local-*` | does the work rank-locally (often misleading — see A6) |
| `TEMP_*.md` | temporary working doc at the repo root |
| `SPIKE_<X>_*.md` | the active optimization program docs |

### Decision-making patterns

- **Reference shape:** 32 slots / 256K context / 256 requests / 64 generated
  tokens per request. Anything tested at smaller shapes is considered
  preliminary, not gating.
- **A/B discipline:** every change ships with a control comparison. No
  "trust me, this is faster."
- **The gate is the gate:** if a change passes the gate, it lands; if it
  fails, it doesn't — regardless of measured perf. The relaxed policy
  shifted the gate from bit-exact-vs-old-binary to agreement-on-quality,
  but the principle is the same.
- **Layered isolation:** smoke binaries reproduce specific issues; the
  full-layer-smoke runs the full serving path. You can pin a regression
  to a sub-binary first, then promote.

### Where the methodology has broken down (and the cleanup sprint fixes it)

- **Flags don't sunset.** The "no permanent semantic variants" rule from
  `AGENT.md` is being violated. The cleanup sprint addresses backlog +
  sets a sunset rule going forward.
- **`TEMP_*.md` accumulate forever.** No retention rule. Cleanup can fold
  this in.
- **The giant smoke is overgrown.** 23k lines in one file is the symptom
  of the flag-matrix problem. The cleanup's rewrite-from-scratch permission
  exists for this.
- **The same identifier means different things in different functions**
  (`rank_major_input` is hardcoded `false` in one function, properly gated
  in another). Cleanup includes catching these.

---

## The one-page mental model

```
   docs/sprints/VISION.md  ◄── direction
                  │
                  ▼
   docs/sprints/SPRINT-NNN.md  ──── per-sprint formal plan
                  │
                  ▼
   tools/ds4-v100-tp-ep-full-layer-smoke.cu  ──── THE 23k-line work surface
                  │     (new code goes here behind --xxx-gate)
                  ▼
   tools/ds4-v100-*-ab.py  ──── A/B harness on the V100 pod
                  │
                  ▼
   TEMP_STATUS_REPORT_NNN.md  ──── written at the repo root, summarizes
                  │
                  ▼
   Decision: promote (flip default, delete flag — should be same commit)
           / reject (delete flag if terminal, keep if revisiting)
           / experimental (tag with sprint owner + sunset)
                  │
                  ▼
   ds4_v100_tp_runtime.cu / .h  ──── the narrow production library
                                     called by ds4-v100-run-appliance.sh
```

Reference shape for any A/B: **32 slots / 256K / 256 req / 64 tok.**
Gate: **agreement-on-quality** (`TEMP_PARITY_POLICY.md`).
Where the noise comes from: **`tools/*.cu` + `TEMP_*.md` + accumulated
unsunset flags.**

## What I'd remove if I were you (and want addressed by the cleanup sprint)

1. The 189 `TEMP_STATUS_REPORT_*.md` files at the repo root — archive them
   under `docs/sprints/archive/` instead of root noise.
2. The ~15 `TEMP_<topic>.md` working docs that are superseded — keep only
   the ones referenced by the in-flight sprint.
3. Per-sprint flags that have been promoted or rejected — the entire point
   of the cleanup sprint.
4. Dead `.cu` smoke binaries that were one-sprint experiments — there are
   several that don't appear to be referenced by any current shell harness.
