# Sprint 229 - TP Runtime Skeleton

Date: 2026-05-23
Status: Complete - TP Runtime Skeleton Allocates Target Arenas

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

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] TP runtime code lives in new TP-specific files.
- [x] No PP scheduler files are modified.
- [x] Smoke tool builds on the V100 pod for `sm_70`.
- [x] Smoke opens all eight GPUs and enables peer access.
- [x] Smoke allocates hidden, KV, and scratch arenas on all eight GPUs.
- [x] Smoke runs a fixture pass and verifies output on all GPUs.
- [x] Smoke reports per-GPU allocation summary.
- [x] Evidence is copied to
      `logs/from-cluster/sprint229-tp-runtime-skeleton/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
make tools/ds4-v100-tp-runtime-smoke
  -> Darwin guard reports CUDA build required
git diff --check
```

V100 build:

```text
cd /workspace/ds4-sprint181
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke
```

V100 default target smoke:

```text
./tools/ds4-v100-tp-runtime-smoke \
  --ctx 262144 \
  --slots 32 \
  --kv-dtype f8 \
  --scratch-mib 1536
```

Result:

```text
tp_runtime_smoke ctx=262144 slots=32 hidden=4096
scratch_bytes=1610612736 fixture_max_abs=0.000000000

per GPU:
  hidden_bytes:      524288
  kv_bytes:      3646642176
  comp_state:    1803550720
  scratch:       1610612736
  total:         7061329920
```

All eight GPUs reported the same allocation and the fixture verified with
`fixture_max_abs=0`. After close, `nvidia-smi` showed `0 MiB` used on all
eight GPUs in the pod, confirming clean teardown.

Evidence is stored in
`logs/from-cluster/sprint229-tp-runtime-skeleton/`.

## Decision

Sprint 229 ships the first separate TP runtime skeleton. It does not execute a
DS4 layer yet, but it proves the TP runtime can own all eight GPUs, enable peer
access, allocate the target 32-slot/256K sharded KV and compression-state
arenas, touch hidden/scratch memory, verify a fixture pass, and release all
GPU allocations cleanly.

Next sprint should implement the first TP dense/KV slice inside this runtime:
resident hidden state plus a bounded DS4 compressed-KV update/read path,
without using the PP scheduler.
