---
sprint: 032
title: V100 Level 2 Base Appliance Usability Gate
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
vision: VISION.md
verdict: SHIP
---

# SPRINT-032: V100 Level 2 Base Appliance Usability Gate

## Overview

Sprint 032 targets the readiness ladder's Level 2: a minimally usable
non-MTP base appliance. The goal is not performance tuning or speculative
decode. The goal is to make the existing one-slot V100 HTTP appliance usable
enough for operator-driven short generation with clear limits, health/status
checks, repeated request evidence, longer decode evidence, and durable logs.

## Outcome Contract

- `SHIP`: the full V100 gate passes, the base HTTP appliance exposes health
  and status, a longer one-slot HTTP generation smoke passes, and readiness
  no longer blocks on Level 2 base usability.
- `EXTEND`: health/status exists but longer HTTP generation or documentation is
  incomplete.
- `STOP`: changes regress selected-token correctness, replay timing, or the
  existing HTTP smoke.

## Non-Goals

- No MTP forward implementation.
- No multi-slot scheduling.
- No throughput optimization claim.
- No production supervisor or cluster deployment controller.
- No OpenAI-compatible API surface beyond the current internal endpoint.

## Implementation

### Phase 1: Server Health And Status

- [x] Keep `/health` as a cheap liveness endpoint.
- [x] Add `/v100/status` and `/status` JSON endpoints with model/index paths,
  context, default token count, max token count, served request count, and
  readiness ladder level.
- [x] Include status output in smoke artifacts.

### Phase 2: Appliance Smoke Hardening

- [x] Have `tools/ds4-v100-appliance-smoke.sh` check `/health` before issuing
  generation requests.
- [x] Have it check `/v100/status`.
- [x] Assert every generation response returns the requested token count.
- [x] Preserve first-token hex validation when an expected hex is supplied.
- [x] Keep repeated request evidence after a single resident upload.

### Phase 3: Longer Decode Gate

- [x] Add a full-gate longer-generation HTTP smoke separate from the existing
  short selected-token HTTP smoke.
- [x] Capture logs under `docs/sprints/drafts/SPRINT-032-*`.
- [x] Keep readiness truthful: Level 2 may pass while MTP remains
  `missing=mtp_forward`.

### Phase 4: Documentation

- [x] Add Sprint 032 report and follow-ups.
- [x] Update the vision readiness ladder current state.
- [x] Commit implementation, docs, and V100 evidence.

## Definition Of Done

- Local build and shell checks pass.
- `tools/ds4-v100-replay --help` documents status endpoints.
- `tools/ds4-v100-appliance-smoke.sh --help` documents health/status behavior.
- On the 8x V100 pod, full gate passes with zero failures.
- On the 8x V100 pod, the longer HTTP generation smoke passes.
- Repo commit records Sprint 032.
