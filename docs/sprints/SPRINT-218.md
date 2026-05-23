# Sprint 218 - 256K Batch Non-Finite Source Isolation

Date: 2026-05-23
Status: Complete

## Overview

Sprint 217 proved that `ctx=262144` active batches above the current
production cap of `16` are not blocked by VRAM. The model opens, launcher
admission can be forced with `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP`, a single
request at `slots=32` works, and worst observed memory stayed near `24.1 GiB`
on 32GB V100s. The real blocker is correctness: concurrent `18`/`20`/`24`/`32`
slot generation fails with HTTP 500, and disabling the output-head fastpath
exposes `output-head logits contained non-finite values`.

Sprint 218 isolates where those non-finite values first appear. This is a
diagnostic sprint on the existing PP/layer serving path. It should produce a
precise layer/stage/slot failure point or prove that HC remains finite until
the output-head path.

## Goals

- Add default-off diagnostics that can check HC tensors for non-finite values
  during batched decode without changing production behavior.
- Reproduce the nearest failing shape, `ctx=262144` and `18` active slots, on
  the V100 pod with production TurboMind/F8 flags.
- Record the first observed non-finite location by phase, layer, stage/GPU,
  slot, token position, tensor index, and value.
- Keep the `256K` production slot cap at `16` unless this sprint also finds and
  validates an obvious localized fix.

## Non-Goals

- No TP scheduler implementation.
- No generic scheduler abstraction.
- No MTP speculative decode work.
- No routed-FFN kernel tuning unless the diagnostic proves the routed path is
  the first non-finite source.
- No `256K` admission-cap promotion based on diagnostics alone.

## Implementation

1. Add a scheduler debug flag, for example `DS4_V100_DEBUG_HC_FINITE=1`.
2. Under that flag only, scan HC tensors after each batched layer transition in
   `ds4_v100_stage_scheduler_decode_hc_layer_span()`.
3. Add a pre-output-head HC finite check for the batch selected-token path so
   the failure can be separated into:
   - layer decode produced non-finite HC;
   - HC stayed finite and output-head/logit projection produced non-finite
     logits.
4. Include enough error detail in the returned HTTP 500 body and server log to
   identify the failing layer/slot/index without needing Nsight first.
5. Add a focused V100 wrapper, for example
   `tools/ds4-v100-256k-finite-gate.sh`, that:
   - runs the same production TurboMind/F8 flags as Sprint 217;
   - uses `ctx=262144`, `slots=18`, `active_microbatch=18`, and short
     generation by default;
   - enables `DS4_V100_DEBUG_HC_FINITE=1`;
   - writes response bodies, server log, `nvidia-smi` samples, and a small
     summary.
6. Run the wrapper on `llm/llamacpp-build-8gpu`, copy logs back under
   `logs/from-cluster/sprint218-256k-finite-source/`, and update docs.

## Parallel Work Lanes

| Lane | Work | Write scope |
|---|---|---|
| A | Scheduler HC finite diagnostics | `ds4_v100_scheduler.c` |
| B | Focused V100 repro wrapper | `tools/ds4-v100-256k-finite-gate.sh` |
| C | Cluster execution and log capture | `logs/from-cluster/sprint218-256k-finite-source/` |
| D | Status/vision/runbook synthesis | sprint/status docs after evidence exists |

Workers are not alone in the codebase. Do not revert unrelated edits. TP work,
if resumed later, must remain a separate codepath; this sprint stays on the
current PP/layer appliance path because that is where the production cap is
currently enforced.

## Files In Scope

| File | Purpose |
|---|---|
| `ds4_v100_scheduler.c` | debug-only HC finite instrumentation |
| `tools/ds4-v100-256k-finite-gate.sh` | repeatable focused failing-shape repro |
| `docs/sprints/SPRINT-218.md` | plan, execution record, decision |
| `docs/sprints/STATUS.md` | topline status |
| `docs/sprints/VISION.md` | vision progress and next lever |
| `docs/operations/DS4-V100-APPLIANCE.md` | operator cap/blocker note if changed |
| `logs/from-cluster/sprint218-256k-finite-source/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before execution evidence is staged.
- [x] `DS4_V100_DEBUG_HC_FINITE=1` instrumentation is default-off and has no
      production-path cost beyond the env check.
- [x] The debug path reports first non-finite HC location or proves HC is
      finite before output-head selection.
- [x] Focused finite gate wrapper exists and passes `bash -n`.
- [x] V100 build passes in `/workspace/ds4-sprint181`.
- [x] V100 `18`-slot/`256K` diagnostic run completes or fails with exact
      phase/layer/slot evidence recorded.
- [x] Logs are copied to
      `logs/from-cluster/sprint218-256k-*`.
- [x] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated if operator guidance changes.
- [x] Changes are committed with explicit `git add` paths.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
base model: /models/DSv4-Flash-256e-fixed.gguf
```

Primary diagnostic shape:

```text
ctx=262144
slots=18
active_microbatch=18
tokens=16-64
requests=18
warmup_requests=0
MTP=off
DS4_V100_DEBUG_HC_FINITE=1
DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=18
```

Escalation shapes:

- `slots=16`, `active_microbatch=16`, `ctx=262144` as the passing control if
  needed.
- `slots=32`, `active_microbatch=32`, `ctx=262144` only after the nearest
  failing shape is characterized.

## Decision Gates

If the first non-finite appears inside layer decode, the next sprint should
target the named layer phase and kernel family directly.

If HC stays finite until output-head selection, the next sprint should target
the batch output-head/logit path instead of attention/KV or FFN kernels.

If the diagnostic changes timing enough to hide the failure, reduce checks to a
binary-search layer interval or add a cheaper device-side finite flag.

## Risks

- Host readback of HC at every layer is intentionally slow and may perturb the
  failing schedule. This is acceptable for first localization, but not for
  performance measurement.
- The first visible non-finite may be downstream of a large finite overflow or
  precision error. If so, a second pass may need magnitude/range probes.
- HTTP 500 bodies may truncate detail; the server log should carry the same
  information.

## Security

No external exposure. No model weights in logs. The debug output should include
only tensor coordinates, scalar values, request slots, and stage/layer
identifiers.

## Dependencies

- Sprint 217 failure evidence for `256K` active batches above 16.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- V100 build pod `llm/llamacpp-build-8gpu`.

## Execution

Implemented default-off HC finite diagnostics in `ds4_v100_scheduler.c`:

- `DS4_V100_DEBUG_HC_FINITE=1` enables the diagnostic family.
- `DS4_V100_DEBUG_HC_FINITE_LAYER_CHECKS=0|1` controls per-layer HC readback.
- `DS4_V100_DEBUG_HC_FINITE_PRE_OUTPUT=0|1` controls the pre-output-head HC
  check.

Added `tools/ds4-v100-256k-finite-gate.sh`, a body-capturing V100 diagnostic
wrapper. It can reproduce the cold path, enable only pre-output checks, enable
full layer checks, and toggle launcher startup warmup.

Local validation:

```text
bash -n tools/ds4-v100-256k-finite-gate.sh
bash -n tools/ds4-v100-run-appliance.sh
```

V100 build validation:

```text
cd /workspace/ds4-sprint181
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

## V100 Evidence

Cold `18`-slot/`256K` reproduction with no HC checks and no startup warmup:

| Ctx | Slots | Startup warmup | Layer checks | Pre-output check | Status 200 | Status other | Max memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 18 | off | off | off | 0 | 18 | `24076 MiB` |

First body:

```json
{"error":"output-head fast selected-token sequence failed"}
```

Pre-output-only diagnostic with the same cold shape:

```text
HC non-finite: phase=pre-output-head stage=7 gpu=7 layer=-1 slot=0 token=4294967295 position=4294967295 index=0 value=nan
```

Full per-layer diagnostic with the same cold shape:

```text
HC non-finite: phase=decode-layer-slot stage=1 gpu=1 layer=6 slot=0 token=0 position=0 index=0 value=nan
```

That localizes the cold failure before the output head. The first visible NaN
appears in stage 1 / GPU 1 at layer 6 during the first token/position path.

The same body-capturing wrapper with launcher startup warmup enabled changes
the result:

| Ctx | Slots | Startup warmup | Status 200 | Status other | Max GPU util | Max memory |
|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 32 | on | 32 | 0 | `89%` | `24124 MiB` |

The warmed production appliance soak then validated the actual target shape:

| Ctx | Slots | Tokens/request | Requests | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 32 | 64 | 32 | `68.631068` | `19.302488` | `67.558707` | 32/32 |

Evidence copied back:

```text
logs/from-cluster/sprint218-256k-finite-none-nowarmup/
logs/from-cluster/sprint218-256k-finite-preoutput-nowarmup/
logs/from-cluster/sprint218-256k-finite-layer-nowarmup/
logs/from-cluster/sprint218-256k-32slot-warmup-none/
logs/from-cluster/sprint218-256k-32slot-warmup-soak/
```

## Decision

Sprint 217's raw failure was real, but it was a cold-start/runtime
initialization failure in the non-warmed path, not proof that `256K`/`32`
cannot serve. The appliance launcher already resolves `DS4_V100_STARTUP_WARMUP`
to enabled when `active_microbatch > 1`; with that warmed path, `32` slots at
`256K` passes correctness and improves practical long-context throughput.

Updated `tools/ds4-v100-run-appliance.sh` so `ctx=262144` admits `32` slots
only when startup warmup resolves enabled. The same config with
`DS4_V100_STARTUP_WARMUP=0` still fails closed at the old cap of `16`.
