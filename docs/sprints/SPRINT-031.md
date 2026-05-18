---
sprint: 031
title: V100 MTP Resident Sidecar Runtime Bridge
status: completed
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../architecture/DS4-V100-LAYOUT.md
report: SPRINT-031-REPORT.md
followups: SPRINT-031-FOLLOWUPS.md
verdict: SHIP
---

# SPRINT-031: V100 MTP Resident Sidecar Runtime Bridge

## Overview

Sprint 031 moves MTP from a parser/readiness contract to a real V100 runtime
object. The appliance can now inspect the DeepSeek-V4 Flash MTP sidecar,
produce typed tensor descriptors, allocate a compact gpu7 device arena, upload
all 32 sidecar tensors, and spot-check resident bytes.

This still does not enable speculative decode. The remaining blocker is the
actual K=1 MTP forward path over Q8_0 dense projections, Q4_K routed experts,
MTP attention/KV state, output-head projection, and draft/verify semantics.

## Outcome Contract

- `SHIP`: MTP sidecar tensors upload into V100 device memory on gpu7, the full
  V100 gate stays green, and readiness advances from `missing=mtp_runtime` to
  `missing=mtp_forward`.
- `EXTEND`: sidecar residency passes only as a standalone command but is not in
  the full gate.
- `STOP`: MTP residency overfills gpu7, corrupts baseline selected-token
  correctness, or regresses HTTP appliance serving.

## Non-Goals

- No MTP draft token generation.
- No speculative accept/reject state machine.
- No Q4_K parity claim for MTP routed experts.
- No MTP batching or multi-slot scheduling.
- No reuse of the main MXFP4 V100 layer-state path for Q4_K MTP tensors.

## Implementation

### Phase 1: Public Sidecar Inventory

**Files:**
- `ds4.h`
- `ds4.c`

**Tasks:**
- [x] Add `ds4_mtp_sidecar_info` and per-tensor descriptor structs.
- [x] Add `ds4_mtp_sidecar_inspect`.
- [x] Preserve `ds4_mtp_sidecar_report` as the report-only wrapper.
- [x] Report compact resident offsets in addition to GGUF offsets.

### Phase 2: V100 Sidecar Runtime Object

**Files:**
- `ds4_v100_mtp.h`
- `ds4_v100_mtp.c`

**Tasks:**
- [x] Own the MTP sidecar fd/mmap independently from the base model.
- [x] Allocate a compact device arena on the selected GPU.
- [x] Upload all 32 validated MTP tensors in chunks.
- [x] Spot-check resident bytes by reading deterministic head/tail slices.
- [x] Expose tensor lookup and arena access for the future K=1 forward pass.

### Phase 3: Tooling And Gate Integration

**Files:**
- `tools/ds4-v100-mtp-residency-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`

**Tasks:**
- [x] Add `tools/ds4-v100-mtp-residency-smoke`.
- [x] Build it with CUDA on Linux and the local arena stub on Darwin.
- [x] Add the residency smoke to the full V100 gate when `--mtp-model` is
  supplied.
- [x] Advance readiness to `missing=mtp_forward` after sidecar validation and
  residency pass.

## Outcome

`SHIP`.

The real MTP sidecar is now device-resident on gpu7:

```text
mtp_runtime	gpu	7
mtp_runtime	arena_kind	device
mtp_runtime	arena_bytes	3807601408
mtp_runtime	uploaded_tensors	32
mtp_runtime	uploaded_bytes	3807600108
mtp_runtime	spot_checks	60
mtp_runtime	free_after_upload_bytes	29937369088
mtp_runtime	PASS	resident_sidecar=1
```

The full V100 gate passes and now reports:

```text
gate	readiness	NOT_READY	missing=mtp_forward
gate	summary	PASS	failures=0 ready=false
```

## Definition Of Done

- The new MTP residency tool builds locally.
- The new MTP residency tool builds with `CUDA_ARCH=sm_70` on the V100 pod.
- The real MTP sidecar uploads to gpu7 device memory with at least 4 GiB
  reserve remaining.
- Full V100 gate passes with `--mtp-model`.
- Selected-token baseline still returns token hex `3136`.
- HTTP appliance smoke still passes two sequential resident requests.
