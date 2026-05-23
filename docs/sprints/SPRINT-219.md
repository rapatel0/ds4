# Sprint 219 - Explicit Warmed Runtime Readiness

Date: 2026-05-23
Status: Planned

## Overview

Sprint 218 changed the practical serving picture: `32` slots at `256K` is
viable on the warmed appliance path, measuring `68.631068` generated tok/s and
`67.558707` continuation tok/s with `32/32` correctness. The cold path remains
unsafe: without startup warmup, the first visible NaN appears in HC at
`stage=1`, `gpu=1`, `layer=6`, `slot=0`, `token=0`, `position=0`.

Sprint 219 removes ambiguity around that warmup dependency. The goal is not to
change model math. The goal is to make runtime readiness explicit,
operator-visible, and covered by the same throughput harnesses used for
admission decisions.

## Goals

- Add an explicit warmed-readiness contract for the production appliance path.
- Make benchmark/admission tooling capable of running the same warmed path as
  `tools/ds4-v100-run-appliance.sh`.
- Re-run `32`-slot/`256K` warmed serving with a longer continuous test and
  record prompt/decode/generated tok/s separately.
- Preserve the cold-path guard: `ctx=262144`, `slots>16` must not launch with
  startup warmup disabled unless an explicit experimental override is used.

## Non-Goals

- No TP scheduler work.
- No MTP speculative decode work.
- No kernel tuning.
- No removal of the existing startup warmup implementation until a replacement
  is validated on the V100 node.

## Implementation

1. Add a warmed-readiness field to `/v100/status` and metrics if it is not
   already visible enough for operators.
2. Update `tools/ds4-v100-sustained-decode-bench.sh` or add a narrow companion
   wrapper so long-running throughput tests use the launcher path, including
   `DS4_V100_STARTUP_WARMUP=auto|1`.
3. Add a focused production gate for the validated target:

   ```text
   ctx=262144
   slots=32
   active_microbatch=32
   tokens=64
   requests>=64
   startup_warmup=auto/on
   MTP=off
   ```

4. Keep a negative launcher check proving `DS4_V100_STARTUP_WARMUP=0` still
   rejects `ctx=262144`, `slots=32` without an experimental cap.
5. If a smaller explicit priming primitive is obvious and safe, add it behind a
   separate flag and compare it against the existing full startup warmup. If it
   is not obvious, keep the full warmup and document that as the production
   readiness mechanism.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-run-appliance.sh` | warmed/cold admission contract |
| `tools/ds4-v100-sustained-decode-bench.sh` | warmed benchmark support if modified |
| `tools/ds4-v100-*-gate.sh` | focused warmed 32-slot/256K production gate |
| `tools/ds4-v100-replay.c` | status/metrics readiness if needed |
| `ds4_v100_replay.c` | explicit priming only if clearly safe |
| `docs/sprints/SPRINT-219.md` | plan, evidence, decision |
| `docs/sprints/STATUS.md` | topline serving mode |
| `docs/sprints/VISION.md` | vision progress |
| `docs/operations/DS4-V100-APPLIANCE.md` | operator command and caveats |
| `logs/from-cluster/sprint219-*` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] Warmed readiness is visible in status/metrics or documented as already
      sufficiently visible.
- [ ] A repeatable warmed production gate exists for `32` slots at `256K`.
- [ ] The warmed gate records generated tok/s, prompt tok/s, continuation tok/s,
      correctness, GPU utilization, and max memory.
- [ ] The warmed `32`-slot/`256K` run passes at `requests>=64`.
- [ ] A negative check proves cold `32`-slot/`256K` launch still fails closed
      without an experimental override.
- [ ] V100 logs are copied to `logs/from-cluster/sprint219-*`.
- [ ] Status, vision, and operations docs are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Verification Strategy

Primary warmed gate:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
ctx=262144
slots=32
active_microbatch=32
tokens=64
requests=64 or 128
startup_warmup=auto/on
```

Negative gate:

```text
DS4_V100_STARTUP_WARMUP=0
DS4_V100_CTX=262144
DS4_V100_SLOTS=32
DS4_V100_ACTIVE_MICROBATCH=32
tools/ds4-v100-run-appliance.sh --check
```

Expected negative result:

```text
DS4_V100_SLOTS=32 exceeds ctx=262144 admission cap 16
```

## Decision Gates

Keep the warmed `32`-slot/`256K` production mode if the longer gate passes
correctness and stays near the Sprint 218 memory envelope.

Do not attempt to replace full startup warmup with a smaller priming primitive
unless the primitive is clearly explained, isolated, and validated against the
same `32`-slot/`256K` gate.

## Risks

- The full startup warmup may hide a deeper lazy initialization bug. That is
  acceptable for production only if readiness is explicit and tested.
- Longer runs may expose fragmentation, queueing, or state-reset bugs that the
  32-request Sprint 218 soak did not hit.
- A direct replay benchmark can still disagree with the production launcher if
  it bypasses startup warmup; this sprint should eliminate that measurement
  mismatch.

## Security

No external exposure. No model weights in logs. Keep logs to request/response
metadata, status, metrics, and GPU utilization.

## Dependencies

- Sprint 218 warmed `32`-slot/`256K` pass and cold-path NaN localization.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- V100 build pod `llm/llamacpp-build-8gpu`.
