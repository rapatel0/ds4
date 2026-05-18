---
sprint: 030
title: V100 MTP Sidecar Readiness Gate
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-030-REPORT.md
followups: SPRINT-030-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-030: V100 MTP Sidecar Readiness Gate

## Overview

Sprint 030 moves MTP from an undifferentiated readiness blocker to a concrete
runtime contract. The appliance now validates the actual DeepSeek-V4 Flash MTP
sidecar GGUF, reports its tensor dtypes/shapes/bytes/kernel families, and wires
that validation into the V100 gate.

This sprint intentionally does not enable speculative decode. Prior MTP work
showed that sidecar loading is not the hard blocker; the risky part is the
Q4_K/Q8_0 MTP forward path and draft/verify state machine. The gate now says
that precisely: `missing=mtp_runtime`.

## Outcome Contract

- `SHIP`: the MTP sidecar validates on the 8x V100 pod, the selected-token and
  HTTP appliance gates stay green, and readiness reports `missing=mtp_runtime`.
- `EXTEND`: the sidecar validates only in a standalone command but is not wired
  into the gate.
- `STOP`: sidecar validation regresses the selected-token baseline or masks MTP
  runtime readiness.

## Non-Goals

- No MTP speculative decode.
- No MTP tensor upload into the V100 resident arenas.
- No MTP Q4_K/Q8_0 kernel parity claim.
- No MTP draft/verify/rollback state machine.
- No multi-slot scheduling.

## Implementation

### Phase 1: Shared Sidecar Validator

**Files:**
- `ds4.h`
- `ds4.c`

**Tasks:**
- [x] Add `ds4_mtp_sidecar_report`.
- [x] Reuse the existing `ds4.c` GGUF parser and `mtp_weights_bind` layout
  validator instead of creating a divergent MTP parser.
- [x] Require `general.architecture=deepseek4_mtp_support`.
- [x] Require exactly the 32 `mtp.0.*` tensors expected by the DS4 MTP block.
- [x] Report dtype, shape, byte count, GGUF offset, and first V100 kernel
  family for each tensor.

### Phase 2: Tooling

**Files:**
- `tools/ds4-v100-mtp-sidecar-gate.c`
- `Makefile`

**Tasks:**
- [x] Add `tools/ds4-v100-mtp-sidecar-gate --mtp-model FILE`.
- [x] Support `--report FILE` for durable cluster artifacts.
- [x] Build the tool through the normal CPU-linked object path.
- [x] Add the tool to `make clean`.

### Phase 3: Gate Integration

**Files:**
- `tools/ds4-v100-gate.sh`

**Tasks:**
- [x] Add `--mtp-model FILE`.
- [x] Build and run the sidecar gate when an MTP model is provided.
- [x] Keep generic `missing=mtp` when no MTP sidecar is supplied.
- [x] Report `missing=mtp_runtime` when the sidecar validates but the runtime
  draft/verify path is still absent.
- [x] Preserve selected-token, replay, and HTTP appliance readiness.

## Outcome

`SHIP`.

The V100 appliance now has a tested MTP sidecar contract. On the cluster, the
sidecar validates as:

```text
architecture=deepseek4_mtp_support
tensors=32
f32: 19 tensors, 7,691,756 bytes
q8_0: 10 tensors, 176,029,696 bytes
q4_k: 3 tensors, 3,623,878,656 bytes
described_tensor_bytes=3,807,600,108
```

The full V100 gate passes and now reports:

```text
gate	readiness	NOT_READY	missing=mtp_runtime
gate	summary	PASS	failures=0 ready=false
```

## Definition Of Done

- `tools/ds4-v100-mtp-sidecar-gate` builds locally and on the CUDA pod.
- The real MTP sidecar validates on the 8x V100 pod.
- Full gate passes with `--mtp-model`.
- The selected-token baseline still returns token hex `3136`.
- The HTTP appliance smoke still passes two sequential resident requests.
- Readiness reports `mtp_runtime`, not generic `mtp`, when the sidecar is
  present and valid.
