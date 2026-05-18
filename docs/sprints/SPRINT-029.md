---
sprint: 029
title: V100 Resident HTTP Appliance Smoke
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-029-REPORT.md
followups: SPRINT-029-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-029: V100 Resident HTTP Appliance Smoke

## Overview

Sprint 029 turns the Sprint 028 replay runtime into a minimal resident process
surface. The goal is not a general chat server yet; it is a deterministic
loopback endpoint that keeps all eight V100 stage schedulers resident, resets
mutable state between independent one-slot requests, and proves the official
selected-token vector over HTTP.

The public artifact is `tools/ds4-v100-replay --serve`, plus
`tools/ds4-v100-appliance-smoke.sh` and gate wiring that measures
`public_serving` readiness.

## Outcome Contract

- `SHIP`: HTTP serving path runs on the 8x V100 pod, returns expected token
  bytes `3136`, and the full gate reports `missing=mtp`.
- `EXTEND`: HTTP compiles but still requires reopening all stages per request,
  or does not become part of the gate readiness.
- `STOP`: reset semantics break the existing selected-token or replay gate.

## Non-Goals

- No OpenAI-compatible API.
- No streaming.
- No concurrent requests.
- No multi-slot scheduler.
- No MTP.
- No startup/upload optimization.

## Implementation

### Phase 1: Reset Semantics

**Files:**
- `ds4_v100_scheduler.h`
- `ds4_v100_scheduler.c`
- `ds4_v100_replay.h`
- `ds4_v100_replay.c`

**Tasks:**
- [x] Add `ds4_v100_stage_scheduler_reset`.
- [x] Reset raw SWA KV, compressed attention KV/state, indexer KV/state, HC
  buffers, and scheduler current-HC pointer.
- [x] Add `ds4_v100_replay_reset` across all eight resident schedulers.
- [x] Clear one-shot runtime state after reset so an HTTP process can serve
  independent sequential prompts without reuploading weights.

### Phase 2: Loopback HTTP Endpoint

**Files:**
- `tools/ds4-v100-replay.c`

**Tasks:**
- [x] Add `--serve`, `--host`, `--port`, and `--max-requests`.
- [x] Serve `GET /health`.
- [x] Serve `POST /v100/selected-token` and
  `POST /v1/v100/selected-token`.
- [x] Parse request JSON for `prompt` and optional `tokens`.
- [x] Reset runtime state before each request.
- [x] Return the same JSON token/timing/memory schema as the CLI replay path.

### Phase 3: Appliance Smoke And Gate

**Files:**
- `tools/ds4-v100-appliance-smoke.sh`
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Add a dependency-light smoke that launches the resident server, waits for
  the serving line, sends two sequential loopback HTTP requests using bash
  `/dev/tcp`, and validates first-token hex on both requests.
- [x] Avoid `curl` and `python3` dependencies because the CUDA pod image is
  intentionally bare.
- [x] Add the HTTP smoke to the full V100 gate.
- [x] Remove `public_serving` from missing readiness only when the HTTP smoke
  passes.

## Outcome

`SHIP`.

The project now has a resident one-slot HTTP appliance surface. It keeps the
8-stage V100 scheduler process alive after upload, resets KV/HC mutation for
each request, and returns the expected official selected token over loopback
HTTP.

The full gate now reports:

```text
gate	readiness	NOT_READY	missing=mtp
gate	summary	PASS	failures=0 ready=false
```

## Definition Of Done

- Reset APIs compile locally and on the CUDA pod.
- `tools/ds4-v100-replay --serve` builds for `sm_70`.
- HTTP smoke returns first-token hex `3136` twice from one resident process.
- Full appliance gate includes the HTTP smoke and passes.
- Full readiness no longer lists `public_serving`; only `mtp` remains.
