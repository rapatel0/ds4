# Sprint 212 - TP4/PP1 Low-Bit Layer Body Pivot

Date: 2026-05-23
Status: Complete

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

- [x] Sprint plan exists.
- [x] New TP4-only low-bit layer-body executable exists.
- [x] No PP scheduler files are modified.
- [x] CUDA target is added to `Makefile`, including macOS CUDA-required branch.
- [x] Local hygiene passes.
- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [x] 16/32/64 tokens-per-active runs pass correctness on four V100s.
- [x] Timing output separates full, TP4 compute, TP4 resident reduce, total,
  and speedups.
- [x] Results are copied to `logs/from-cluster/sprint212-tp4-lowbit-layer/`.
- [x] Sprint 212 document records validation and decision.
- [x] Status/Vision documents are updated.
- [x] Changes are committed with explicit `git add` paths.

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

## Implementation Notes

Added `tools/ds4-v100-tp4-turbomind-layer-smoke.cu` as a separate TP-only
tool. It deliberately does not share scheduler abstractions with the PP/layer
runtime. The tool uses the public TurboMind ABI through `dlopen`, builds
deterministic MXFP4 fixtures, runs a full one-GPU reference and four TP4
middle-dimension shards on GPUs `0,1,2,3`, and reports resident root-reduce
timing separately from TP4 compute timing.

Two implementation bugs were found and fixed during V100 validation:

- the forked TP8 fixture exponent range could overflow the TP4 synthetic
  outputs; it now uses the known-good TP4 control range;
- `run_side()` must call `cudaSetDevice(side.device)` before invoking the
  TurboMind ABI. Without that, finite but wrong shard outputs were produced.

The control `test_ggml_turbomind_tp_split_4gpu` was also rebuilt and rerun on
the same pod/library to confirm that the Sprint 202 TP4 correctness baseline
still passes.

## Validation

V100 pod: `llm/llamacpp-build-8gpu`, node `gpu-01`.

Build:

```text
make -j80 tools/ds4-v100-tp4-turbomind-layer-smoke CUDA_ARCH=sm_70
```

Results:

| Tokens/active expert | Routes | Full reference | TP4 compute | Resident reduce | TP4 total | Compute speedup | Total speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 96 | `0.292096 ms` | `0.125082 ms` | `0.145972 ms` | `0.271054 ms` | `2.335x` | `1.078x` | pass |
| 32 | 192 | `0.346266 ms` | `0.133325 ms` | `0.238310 ms` | `0.371635 ms` | `2.597x` | `0.932x` | pass |
| 64 | 384 | `0.602675 ms` | `0.162560 ms` | `0.461001 ms` | `0.623561 ms` | `3.707x` | `0.967x` | pass |

Evidence is stored in
`logs/from-cluster/sprint212-tp4-lowbit-layer/`.

## Decision

Reject TP4/PP1 runtime ownership as the next sprint.

The TP4 low-bit compute body is correct and compute-only speedup is strong, but
the resident root reduction is not a material serving win. It clears the
96-route case by only `1.078x`, then regresses at 192 and 384 routes. The prior
TP4 control remains worse once copy-inclusive payloads are included.

This is enough evidence to keep TP4 as a future prefill/larger-batch candidate,
but not enough to build a TP4 serving runtime. Sprint 213 should return to a
monolithic/persistent low-bit routed-FFN executor or implement a genuinely
better TP collective/fused-reduction primitive before revisiting TP runtime
ownership.
