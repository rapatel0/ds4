# Sprint 577 - C1 Full-Capture Instability: Scaling + Sanitizer Localization

Date: 2026-05-29

## Goal

Localize the full-capture batch-instability defect from Sprint 576: is it a
static pointer/buffer bug (constant in slot count) or accumulation-order
nondeterminism (scales with active routed tokens)? Use slot-count scaling and
compute-sanitizer.

## Result summary

The instability **scales with the number of active routed tokens** and is ~zero
at 1-2 active slots, which argues against a static pointer/buffer bug and points
at accumulation-order nondeterminism in the MoE route/compose path under graph
replay. compute-sanitizer cannot reach the bug region on this appliance (OOMs
before decode; no low-memory repro exercises the graph-replay path), so the
decisive confirm-and-fix is a deterministic-compose rebuild.

## Scaling discriminator (full-vs-full logit floor by active slot count)

Logit-space full-vs-full floor (two identical full-capture runs), `config=8`
where applicable, warmup disabled, top-1 logit per token via the
`tp_ep_decode_top1_logit` diagnostic:

| active slots | matched-token \|logit Δ\| (max / mean) | token flips |
| ---: | --- | ---: |
| `1` | bit-exact | `0` |
| `2` | `0.25` / `0.0095` | `0/2` |
| `8` | `1.21` / `0.039` | `8/8` |
| `32` | `3.63` / `0.44` | `32/32` |

The 8-slot graph structure is identical across these runs; only the number of
active routed tokens changes. Instability is ~zero at 1-2 active and grows
continuously with active count. A static pointer/buffer-structure bug would be as
unstable at 2 active as at 8; it is not. So the defect tracks **active route
count**, consistent with nondeterministic accumulation order in the route/compose
path (the `atomicAdd` combine in `kernels/v100/compose.cuh` / `router.cuh`),
exposed by graph replay. (`N=4` and a `SLOTS=16` cell failed to produce data on
relaunch -- an intermittent second-launch issue -- but `N=2` and `N=8` bracket
the onset and the conclusion does not depend on them.)

The open reconciliation: eager uses the same atomics and is bit-stable
(eager-vs-eager Δ = 0, Sprint 576), so eager's accumulation order is effectively
deterministic for this shape. The graph-replay path is what makes it
nondeterministic -- most likely by changing the serialization/concurrency of the
per-rank compose that eager performs in a fixed order.

## compute-sanitizer attempt (initcheck)

Ran `compute-sanitizer --tool initcheck` wrapping the appliance at a reduced
shape (`slots=8`, `position=2048`, `6` tokens, full-capture gates), via a
run-script variant that prepends a sanitizer prefix to the exec.

Outcome:

- The sanitizer initialized NCCL and instrumented the full 8-GPU startup (dense
  load, rank buffers, NCCL collectives, tp_runtime -- all PASS), then **OOMed
  during expert weight loading** (`turbomind_bindings.cu:153: out of memory`,
  ~layer 41). initcheck shadow memory plus the ~24 GB/GPU model exceeds the
  32 GB cards. It never reached the decode/compose path where the bug lives.
- `ERROR SUMMARY: 0 errors` for everything it did instrument (startup / NCCL /
  dense / partial expert load).

Conclusions about the tool for this bug:

- The memory overhead is the hard blocker at this scale (predicted, now
  confirmed). NCCL/8-GPU under the sanitizer otherwise works.
- No existing smoke test (`tests/*smoke*`) exercises the cudagraph capture/replay
  path; they are eager-kernel tests. So there is no low-memory vehicle that
  reproduces this graph-replay-specific bug for the sanitizer.
- Even with memory, the leading mechanism (FP atomic-accumulation order) is not
  an error class compute-sanitizer detects; it would report clean. The attempt
  did usefully eliminate gross uninitialized/OOB errors in the startup/load path.

## Decision

The instability is a determinism defect in the graphed MoE route/compose
accumulation, scaling with active routes. compute-sanitizer is not a viable tool
for it here (OOM + no graph-replay repro + wrong error class).

Next (Sprint 578): the deterministic-compose rebuild. Flip the captured-region
compose to a deterministic, fixed-order reduction (e.g.
`nccl_reduce_scatter_compose_gate=true` or a non-atomic combine; these are
compile-time defaults in `engine/runtime_options.cuh` with no CLI toggle, so a
rebuild is required), rerun the full-vs-full logit floor at `8`/`32` slots, and
check whether full's floor collapses toward the eager floor (Δ -> 0). If it does,
the mechanism is confirmed and the determinism defect is fixed; full capture can
then be re-evaluated for promotion against the eager floor.

## Definition of Done

- Scaling discriminator recorded (full-vs-full floor at 1/2/8/32 active slots).
- compute-sanitizer attempt recorded with the OOM blocker and the 0-error load
  path.
- Mechanism conclusion (accumulation-order under graph replay) and the
  deterministic-compose rebuild plan recorded.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.
