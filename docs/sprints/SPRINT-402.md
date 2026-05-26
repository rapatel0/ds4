# Sprint 402: NCCL VRAM Admission Guard

Date: 2026-05-26

## Overview

Sprints 400 and 401 proved that narrow serving-facing NCCL gates are
functionally correct at smaller shapes but not target-shape admitted at
`32` slots / `256K`. Both gates add roughly `0.6-0.7 GiB/GPU` of resident NCCL
communicator memory and then OOM later during raw-SWA allocation, despite the
generic `64 MiB` VRAM guard passing at startup checkpoints.

This sprint makes that lesson executable. NCCL gates should have an explicit
minimum-free-VRAM admission threshold so doomed configurations fail before
decode begins, with a clear artifact reason.

## Constraints

- TP/EP only. No PP/layer-split work.
- No new NCCL performance gate.
- Preserve existing non-NCCL defaults.
- Keep NCCL gates default-off.
- The guard must be configurable from launcher, direct profile harness, and
  the full-layer smoke binary.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`

Planned changes:

1. Add `--nccl-min-free-mib N` to the full-layer smoke.
2. Add an `nccl_gate_active()` helper covering current NCCL gates:
   reduce-scatter compose, HC-current allgather, and attention-output
   allgather.
3. After rank buffers and NCCL communicator initialization, enforce the
   NCCL-specific threshold with `tp_ep_vram` / `tp_ep_vram_summary` rows.
4. After output-head allocation, enforce the same threshold before decode.
5. Add launcher env `DS4_V100_TP_EP_NCCL_MIN_FREE_MIB`, defaulting to `1536`
   when any NCCL gate is active and `0` otherwise.
6. Add profile flag `--nccl-min-free-mib`.

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

V100 admission checks:

1. Non-NCCL control at `32` slots / `256K` with `--nccl-min-free-mib 1536`
   should ignore the NCCL threshold and pass the normal path.
2. NCCL HC-current candidate at `32` slots / `256K` with
   `--nccl-min-free-mib 1536` should fail before decode with an explicit
   NCCL VRAM admission failure, rather than OOMing during raw-SWA allocation.
3. NCCL HC-current candidate at `16` slots / `256K` with
   `--nccl-min-free-mib 1536` should remain admitted.

## Results

Local checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100 build passed:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Admission matrix:

| Case | Slots / ctx | NCCL gate | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | NCCL admission |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Control | `32 / 256K` | off | `0` | `54639` | `93.773792` | `104.390813` | `1746 MiB` | ignored |
| HC-current NCCL | `32 / 256K` | `--tp-hc-current-input-nccl-allgather-gate` | `14` | n/a | n/a | n/a | `1114 MiB` | failed at `1536 MiB` threshold |
| HC-current NCCL | `16 / 256K` | `--tp-hc-current-input-nccl-allgather-gate` | `0` | `54639` | `63.523008` | `72.165002` | `3820 MiB` | passed |

The rejected 32-slot NCCL candidate now stops at:

```text
tp_ep_nccl_vram_admission_failed label=nccl_after_output_head min_free_mib=1536
```

The corresponding artifact has:

```text
vram_nccl_after_output_head_min_free_mib=1114
vram_nccl_after_output_head_threshold_mib=1536
vram_nccl_after_output_head_failures=5
returncode=14
```

Artifacts:

- `logs/from-cluster/sprint402-nccl-vram-admission/direct-control/`
- `logs/from-cluster/sprint402-nccl-vram-admission/direct-nccl-fail/`
- `logs/from-cluster/sprint402-nccl-vram-admission/direct-nccl-pass-16/`

## Decision

Promote the NCCL admission guard as permanent harness behavior. Keep NCCL
serving gates default-off and diagnostic-only. The guard does not improve
throughput, but it prevents repeating expensive target-shape NCCL runs that
already fall below the empirically required VRAM reserve.

The immediate performance implication is unchanged from Sprints 400-401:
narrow NCCL replacement at individual serving boundaries is not sufficient.
Future NCCL work needs a broader memory-planned TP/EP boundary that replaces
peer-copy traffic and avoids paying isolated communicator overhead on top of
the current resident layout.

## Definition of Done

- NCCL-specific admission threshold exists in binary, launcher, and profile
  harness.
- V100 build passes.
- V100 artifacts prove the 32-slot NCCL candidate fails early for admission,
  and the 16-slot candidate is still admitted.
- Sprint doc, temporary status report, status, and vision are updated.
- Commit all kept artifacts explicitly.

## Risks

- The exact reserve is empirical. `1536 MiB` is chosen because the rejected
  NCCL target runs reached only `1114 MiB` free, while non-NCCL control had
  `1746 MiB`.
- This is a guardrail sprint, not a throughput improvement. Its value is to
  prevent repeated expensive OOM runs while preserving the path to broader
  memory-planned NCCL work.
