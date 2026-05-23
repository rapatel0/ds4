# Sprint 229 - TP Runtime Skeleton

Date: 2026-05-23
Status: Planned

## Overview

Sprint 228 emitted the TP/EP pack contract. The next step is a separate TP
runtime skeleton that proves lifecycle and residency without touching the
frozen PP scheduler.

This sprint creates the first TP-owned runtime files. The runtime opens all
eight V100s, enables peer access, allocates resident hidden/KV/scratch arenas
from the target TP contract, executes a fixture pass, reports per-GPU
allocation, and closes cleanly.

## Goals

- Add separate TP runtime skeleton files.
- Add a smoke tool that exercises the skeleton on all eight GPUs.
- Allocate resident arenas for:
  - hidden input/output;
  - sharded KV/cache bytes;
  - scratch bytes.
- Use the target default shape:
  - slots=32;
  - ctx=256K;
  - hidden=4096;
  - KV dtype F8;
  - TP=8;
  - EP=8.
- Run a fixture pass that touches hidden and scratch memory on every GPU.
- Report per-GPU allocation bytes and verify fixture output.

## Non-Goals

- No PP scheduler edits.
- No generic PP/TP abstraction.
- No real DS4 layer execution.
- No pack loader.
- No MTP.
- No throughput claim.

## Implementation

1. Add `ds4_v100_tp_runtime.h`.
2. Add `ds4_v100_tp_runtime.cu`.
3. Add `tools/ds4-v100-tp-runtime-smoke.cu`.
4. Add Makefile targets.
5. Runtime API:
   - open config;
   - enable peer access;
   - allocate per-GPU arenas;
   - run fixture pass;
   - report allocation summary;
   - close/free.
6. Build locally only far enough to verify Makefile guards on non-CUDA hosts.
7. Build and run on the V100 pod with:
   - default target allocation;
   - a bounded small allocation if full target allocation is blocked by
     transient memory pressure.
8. Copy evidence to
   `logs/from-cluster/sprint229-tp-runtime-skeleton/`.

## Files In Scope

| File | Purpose |
|---|---|
| `ds4_v100_tp_runtime.h` | TP runtime skeleton API |
| `ds4_v100_tp_runtime.cu` | TP runtime skeleton implementation |
| `tools/ds4-v100-tp-runtime-smoke.cu` | V100 smoke tool |
| `Makefile` | build targets |
| `docs/sprints/SPRINT-229.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint229-tp-runtime-skeleton/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] TP runtime code lives in new TP-specific files.
- [ ] No PP scheduler files are modified.
- [ ] Smoke tool builds on the V100 pod for `sm_70`.
- [ ] Smoke opens all eight GPUs and enables peer access.
- [ ] Smoke allocates hidden, KV, and scratch arenas on all eight GPUs.
- [ ] Smoke runs a fixture pass and verifies output on all GPUs.
- [ ] Smoke reports per-GPU allocation summary.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint229-tp-runtime-skeleton/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Decision

Pending.
