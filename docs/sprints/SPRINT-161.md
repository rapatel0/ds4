# Sprint 161 - Small-Route Fused Executor Serving Probe

Date: 2026-05-21

## Objective

Find a practical middle ground between the fast per-slot stage pipeline and the
slow `16`-slot global chunk path.

Sprint 160 proved that `DS4_V100_ASYNC_SLOT_CHUNK=16` exposes the desired
`96`-route routed-FFN shape, but it also stalls downstream stages until the
whole chunk finishes. The next useful serving experiment is therefore not a
bigger chunk. It is a smaller fused executor shape that can run with `chunk=2`
or `chunk=4`, preserving more pipeline overlap while still giving TurboMind a
denser route group than a single request.

Primary target:

```text
ctx = 262144
slots = 16
active_microbatch = 16
async_slot_chunk = 2
routes = 2 * 6 = 12
```

Secondary target:

```text
async_slot_chunk = 4
routes = 4 * 6 = 24
```

This secondary target is a probe only. If it does not survive a serving smoke,
do not keep it exposed in the runtime.

## Rationale

The current 256K sustained baseline is about `63-64` generated tok/s with
per-step async and one-slot scheduler calls. Whole-stage `chunk=2` was close
but slower on the short A/B, while `chunk=4+` regressed hard. A fixed 12-route
or 24-route gated-SiLU executor can test whether the small chunk path is
limited by the generic gate/up kernel rather than by the stage barrier alone.

This is intentionally narrower than a TP/EP rewrite. If small-route fusion does
not move the 256K serving curve, the next material implementation should pivot
to topology or a true layer-level wavefront scheduler.

## Initial Scope

- Add a TurboMind exported DS4 gated-SiLU probe function for `12` total routes
  using the existing V100 `M16` kernel configuration.
- Add runtime lookup and dispatch through
  `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed12`.
- Add launcher validation aliases:
  - `fixed12`, `chain12`, `ffn12`, `12`
- Keep all new paths opt-in.
- Benchmark at the actual practical target: `ctx=262144`, `slots=16`.

## Non-Goals

- No production default change unless throughput improves materially and
  correctness remains exact.
- No broad tensor-parallel scheduler rewrite.
- No MTP enablement.
- No 32K-only success criterion.

## Implementation Plan

1. Add a `12` route TurboMind probe export.
2. Wire the new function pointers through `ds4_cuda.cu`.
3. Extend routed-executor env parsing in the launcher and runtime.
4. Build the TurboMind library and V100 replay binaries on the cluster.
5. Run focused correctness smoke with the new executor mode.
6. Run 256K serving A/B:
   - control per-step chunk default;
   - `chunk=2 + fixed12`;
   - optional `chunk=4 + fixed24` only if the `24` route probe survives a
     short serving smoke.
7. Copy logs to `logs/from-cluster/`.
8. Record the decision in this sprint document and `docs/sprints/VISION.md`.

## Definition Of Done

- `git diff --check` passes.
- Launcher syntax check passes.
- Launcher `--check` accepts `fixed12` if the route remains shippable, or
  rejects it again if validation shows it should not be exposed.
- V100 build passes for the touched DS4 runtime and TurboMind library.
- A full scheduler or served smoke proves the selected executor is actually
  selected for `total_routes=12`, or records why it was removed.
- 256K served correctness remains exact for each tested mode.
- Generated and continuation/decode tok/s are recorded separately.
- A default-promotion decision is made from measured 256K evidence.

## Decision Gate

Promote only if the 256K continuation/decode rate beats the current sustained
baseline (`~62-64 tok/s`) outside normal run noise and the new path does not
increase memory pressure enough to threaten 32 GiB V100 residency.

If both small-route fused paths are flat or slower, keep the default unchanged
and move the next sprint to TP/EP or an in-stage layer-wavefront scheduler.

## Results

Implementation attempt:

- Added and built a `12`-route TurboMind gated-SiLU probe on the cluster.
- Wired `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed12` temporarily for served
  testing.
- Also tried a `24`-route variant during execution, but it did not survive a
  short serving smoke and was removed from the exposed runtime contract.

Validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed12 ... --check
git diff --check
cmake --build build/turbomind-v100-s127 -j80
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay tests/cuda_v100_stage_scheduler_smoke
```

The temporary `fixed12` path did select in served mode:

```text
ds4: TurboMind routed executor fixed12 shape total_routes=12 active_experts=6 max_routes_per_expert=2
ds4: TurboMind routed executor selected fixed gate_up total_routes=12
```

256K serving A/B:

| Mode | Generated tok/s | Continuation tok/s | Correctness | Result |
|---|---:|---:|---:|---|
| `chunk=2`, generic routed executor | `60.879872` | `59.928624` | `16/16` | control |
| `chunk=2 + fixed12` | `60.709924` | `59.761332` | `16/16` | selected but flat/slower |
| `chunk=16`, layer batch disabled | `16.201644` | `15.948493` | `16/16` | confirms stage-wide barrier dominates |
| `chunk=4 + fixed24` short smoke | no summary | no summary | n/a | wedged in startup warmup; removed |
| final rebuilt `chunk=2 + fixed12` | no summary | no summary | n/a | selected, then wedged in startup warmup; removed |

All throughput runs used:

```text
ctx = 262144
slots = 16
active_microbatch = 16
tokens = 64
requests = 16
async_pipeline_mode = per-step
async_event_handoff = true
MTP = off
```

## Decision

Do **not** ship the small-route executor path.

The one successful `fixed12` served run proved dispatch and correctness, but it
was slightly slower than the same-build `chunk=2` control and the final rebuilt
binary wedged in startup warmup after selecting the fixed path. The temporary
runtime and TurboMind API changes were removed before commit.

This closes the small-route chunk hypothesis:

- `chunk=16` is not worth tuning; the stage-wide barrier dominates even when
  full-layer batching is disabled.
- `chunk=2` preserves more pipeline overlap, but a fixed 12-route gate/up
  probe does not improve throughput.
- The next material implementation should be either:
  - a true in-stage layer-wavefront scheduler that batches same-layer FFN work
    without delaying stage handoff for the whole slot group; or
  - a bounded TP/EP prototype that creates denser kernels without relying on
    slot chunking.

Artifacts:

- `logs/from-cluster/sprint161-chunk16-no-layer-batch-16slot-64tok-16req/`
- `logs/from-cluster/sprint161-chunk2-control-16slot-64tok-16req/`
- `logs/from-cluster/sprint161-fixed12-chunk2-16slot-64tok-16req/`
- `logs/from-cluster/sprint161-fixed24-chunk4-smoke-16slot-2tok-1req/`
- `logs/from-cluster/sprint161-final-fixed12-chunk2-16slot-64tok-16req/`
