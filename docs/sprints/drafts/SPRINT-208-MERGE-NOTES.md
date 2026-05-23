# Sprint 208 Merge Notes

Date: 2026-05-23

## Planning Inputs

- User direction: TP should be a completely separate investigation/runtime path,
  with new files as the default.
- Architecture note: `docs/architecture/DS4-V100-TP8-INVESTIGATION.md`.
- Existing evidence: Sprints 201-205 TP4 boundary/compute/resident tests and
  Sprint 206 six-route boundary result.
- New planner starting point: `tools/ds4-v100-plan-tp.c`.

## Accepted Scope

- Plan a TP8 investigation sprint, not production TP8 serving.
- Preserve current PP/layer scheduler as baseline/control.
- Require concrete V100 evidence: TP8 collectives, resident compute between
  reductions, NVLink traffic/counter evidence, and KV-sharding memory gates.
- Make new TP files the norm: planner, probes, pack descriptors, and tests.
  Do not implement any scheduler, generic or TP-only, in Sprint 208.

## Rejected Scope

- Retrofitting `ds4_v100_scheduler.*` to handle TP8.
- Creating a generic scheduler abstraction.
- Creating a TP scheduler skeleton before the planner/probe gates run.
- Making TP8 a launcher default.
- Combining TP8 with MTP token commit.
- Using replicated KV as a viable 32-slot/256K design.

## Key Synthesis

The strategic insight is that `PP1/TP8` has no obvious byte-volume downside at
the 32-slot target. The estimated TP wire is tens of MiB per decode step, which
is small compared with V100 NVLink bandwidth. The risk is synchronization and
ownership complexity: many small collectives, sharded KV correctness, and
balanced resident execution across all eight GPUs.

Therefore Sprint 208 should not start by integrating a scheduler. It should
build the TP-specific evidence ladder and stop if the 32-slot/256K resident
boundary is not competitive.
