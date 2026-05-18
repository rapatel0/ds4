---
sprint: 028
title: V100 Replay Runtime And Timing Tool
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-028-REPORT.md
followups: SPRINT-028-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-028: V100 Replay Runtime And Timing Tool

## Overview

Sprint 028 extracts the working V100 selected-token path out of the smoke test
shape and into a reusable one-shot replay runtime. The runtime owns the
8-stage scheduler array, model mapping, tokenizer metadata, prompt replay,
greedy continuation, and timing counters.

The public artifact for this sprint is `tools/ds4-v100-replay`: a CUDA-only
appliance command that can replay a prompt on the 8x V100 stack, generate
greedy tokens, verify expected token bytes, and emit JSON timing/memory data.

## Outcome Contract

- `SHIP`: reusable replay runtime builds on `sm_70`, command tool returns the
  expected first token bytes `3136`, and the gate records timing counters.
- `EXTEND`: selected-token correctness remains available only through the test
  smoke, or timing data is not machine-readable.
- `STOP`: extraction breaks the existing scheduler correctness gate.

## Non-Goals

- No HTTP server yet.
- No multi-request scheduler reset.
- No MTP.
- No multi-slot scheduling.
- No FP8 KV baseline.

## Implementation

### Phase 1: Reusable Replay Runtime

**Files:**
- `ds4_v100_replay.h`
- `ds4_v100_replay.c`

**Tasks:**
- [x] Open the CPU inspect-only tokenizer engine.
- [x] Map the source model once and register the model fd for CUDA.
- [x] Open all eight V100 stage schedulers with the F16 KV default.
- [x] Replay prompt tokens through stage 0, cross-GPU handoffs, and stages 1-7.
- [x] Generate greedy continuation tokens by feeding selected tokens back into
  the scheduler.
- [x] Track open/upload time, stage decode time, handoff time, output-head
  time, token text time, uploaded bytes, arena bytes, and layer executions.

### Phase 2: Appliance Command

**Files:**
- `tools/ds4-v100-replay.c`
- `Makefile`

**Tasks:**
- [x] Add `tools/ds4-v100-replay`.
- [x] Support `--model`, `--index`, `--prompt`, `--prompt-file`, `--ctx`,
  `--tokens`, `--expected-token-hex`, and `--json`.
- [x] Emit token id, text, text hex, logits, timing, tok/s, upload, and arena
  data as JSON.
- [x] Keep the target CUDA-only on Linux and fail clearly on Darwin.

### Phase 3: Gate Wiring

**Files:**
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Build `tools/ds4-v100-replay` in the V100 appliance gate.
- [x] Run a two-token greedy replay against `short_reasoning_plain`.
- [x] Keep `public_serving` missing until an HTTP/process serving surface
  exists.
- [x] Remove `throughput_benchmark` from missing readiness once replay timing
  passes.

## Outcome

`SHIP`.

The project now has a reusable V100 one-slot replay runtime and a scriptable
command-line appliance surface. This is not yet a deployed HTTP server, but it
is the first non-test path that loads the 8-stage V100 scheduler, produces
tokens, and records timing.

## Definition Of Done

- Replay runtime and tool compile locally as C objects.
- `tools/ds4-v100-replay` links and runs on the 8x V100 pod.
- The first generated token for `short_reasoning_plain` matches expected bytes
  `3136`.
- JSON output includes timing and memory counters.
- Full appliance gate includes the replay tool and reports readiness with
  throughput satisfied.
