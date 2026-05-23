# Sprint 212 - TP4/PP1 Low-Bit Layer Body Pivot

Date: 2026-05-23
Status: Planned

## Overview

Pivot from the rejected TP8 MXFP4 expert shape to a bounded TP4/PP1 low-bit
layer-body smoke in separate TP-only files.

Sprint 211 showed that the current TurboMind MXFP4 path is correct at
`mid_shard=512` in the existing TP4 control, but not at TP8
`mid_shard=256`. Sprint 212 should turn that control into a repo-owned
`tools/` layer-body gate with explicit resident reduction timing and practical
route shapes.

## Rationale

Sprint 211 result:

| Topology | Shard | Correctness | Compute signal | Total signal |
|---|---:|---|---:|---:|
| TP8 | `mid_shard=256` | FAIL, NaNs | `3.927x-4.189x` | `0.317x-0.524x` |
| TP4 control | `mid_shard=512` | PASS | `2.333x-3.676x` | `0.629x-0.778x` |

This means the immediate low-bit path should not be TP8 serving integration.
The next useful implementation is TP4/PP1 low-bit layer ownership, using the
known-correct `mid_shard=512` expert split and measuring whether a better
resident reduction boundary can improve the total signal.

## Scope

1. Add `tools/ds4-v100-tp4-turbomind-layer-smoke.cu`.
2. Use public TurboMind C ABI through `dlopen`.
3. Use deterministic synthetic MXFP4 fixtures.
4. Run a full reference and four TP4 shards on one NVLink island.
5. Reduce four partial hidden outputs on device, not only via host-side
   copy-inclusive timing.
6. Run `tokens_per_active=16`, `32`, and `64`.
7. Report:
   - full reference time;
   - TP4 compute time;
   - TP4 resident reduce time;
   - TP4 total time;
   - compute and total speedups;
   - correctness.
8. Decide whether Sprint 213 should build TP4/PP1 runtime ownership or return
   to monolithic persistent routed-FFN kernels.

## Non-Goals

- No generic scheduler.
- No PP scheduler changes.
- No `ds4_v100_scheduler.*` changes.
- No launcher defaults.
- No full serving integration.
- No model weights in logs.
- No cleanup or commit of unrelated Sprint 207 dirty runtime files.

## Architecture

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp4-turbomind-layer-smoke.cu` | TP4 low-bit layer-body smoke |
| `logs/from-cluster/sprint212-tp4-lowbit-layer/` | V100 evidence |
| `docs/sprints/SPRINT-212.md` | Plan and outcome |

The tool may reuse the Sprint 211 packing helpers, but constants and reporting
should be TP4-specific to avoid a generic scheduler abstraction.

## Definition Of Done

- [ ] Sprint plan exists.
- [ ] New TP4-only low-bit layer-body executable exists.
- [ ] No PP scheduler files are modified.
- [ ] CUDA target is added to `Makefile`, including macOS CUDA-required branch.
- [ ] Local hygiene passes.
- [ ] V100 build passes with `CUDA_ARCH=sm_70`.
- [ ] 16/32/64 tokens-per-active runs pass correctness on four V100s.
- [ ] Timing output separates full, TP4 compute, TP4 resident reduce, total,
  and speedups.
- [ ] Results are copied to `logs/from-cluster/sprint212-tp4-lowbit-layer/`.
- [ ] Sprint 212 document records validation and decision.
- [ ] Status/Vision documents are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Decision Gate

Continue TP4/PP1 implementation if:

- correctness passes at all required route shapes;
- resident reduction improves total signal materially versus the existing TP4
  copy-inclusive control;
- memory/topology implications still support 32-slot/128K-256K practical use.

If TP4 remains total-slower even with resident reduction, pause TP runtime work
and return to a monolithic/persistent low-bit routed-FFN executor.

## Risks

- The tool may reproduce the existing TP4 result without improving the
  reduction boundary.
- TP4/PP1 may fit memory but still give up too much topology simplicity versus
  layer-split serving.
- This does not yet include attention/KV or output head ownership.

## Security

No service exposure. Synthetic fixtures only. No model weights in logs.

## Dependencies

- Sprint 211 TP8 rejection and TP4 control evidence.
- Existing TurboMind V100 build.
- V100 build pod or direct node access.
