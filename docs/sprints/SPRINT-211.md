# Sprint 211 - TP8 TurboMind MXFP4 Expert Body

Date: 2026-05-23
Status: Rejected

## Overview

Replace the Sprint 210 FP16 fixture body with a bounded TP8 TurboMind MXFP4
routed-FFN body in a completely separate TP-only executable.

This sprint should answer whether the real low-bit expert layout and kernels
still support the TP8 direction. It is not a serving integration sprint.

## Rationale

Sprint 210 showed that useful FP16 Tensor Core layer-shaped work can live
inside the TP8 boundary:

| Shape | Total avg | Fixture TFLOP/s | Correctness |
|---|---:|---:|---|
| `mid_shard=1024`, 32 tokens | `0.614750 ms` | `10.480` | ok |
| `mid_shard=1024`, 128 tokens | `0.796927 ms` | `32.336` | ok |
| `mid_shard=2048`, 128 tokens | `0.818660 ms` | `62.956` | ok |

That proves topology and resident compute shape, but DS4 experts are not FP16
weights. The project target is low-bit source quantization with GPU-side
unpack/dequant feeding V100 Tensor Core math. The existing TurboMind MXFP4
kernels are the closest real implementation path.

The next gate is therefore an eight-GPU TP split of a DS4-shaped MXFP4 expert
FFN:

```text
full reference on GPU0:
  MXFP4 gated gate/up -> MXFP4 down

TP8 candidate:
  each GPU owns mid_shard = full_mid / 8
  MXFP4 gated gate/up shard
  MXFP4 down shard
  reduce eight partial hidden outputs

compare:
  sum(TP8 partials) ~= full reference
```

## Scope

1. Add `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu`.
2. Use the public TurboMind C ABI through `dlopen`.
3. Use deterministic synthetic MXFP4 fixtures, not model weights.
4. Pack full-width and TP8-sharded gate/up/down expert weights.
5. Run a full single-GPU reference on GPU0.
6. Run eight TP participants across all V100s.
7. Reduce/sum the eight TP down outputs to GPU0 for correctness and timing.
8. Measure:
   - full reference compute time;
   - TP8 max compute time;
   - TP8 reduce/copy time;
   - total TP8 time;
   - compute and total speedup.
9. Test practical routed shapes:
   - `tokens_per_active=16`, `routes=96`;
   - `tokens_per_active=32`, `routes=192`;
   - optional `tokens_per_active=64`, `routes=384`.

## Non-Goals

- No generic scheduler.
- No PP scheduler changes.
- No `ds4_v100_scheduler.*` changes.
- No launcher defaults.
- No full model serving.
- No pack-file format changes.
- No model weights in logs.
- No attempt to clean up or commit unrelated Sprint 207 dirty runtime files.

## Architecture

New files:

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu` | TP-only MXFP4 expert FFN executable |
| `logs/from-cluster/sprint211-tp8-turbomind-ffn/` | V100 build/run evidence |
| `docs/sprints/SPRINT-211.md` | Sprint plan and outcome |

The executable may copy helper patterns from TurboMind tests, but it should not
modify TurboMind API or scheduler code in this sprint.

### Precision Contract

Use `GGML_TM_DTYPE_MXFP4` fixtures with group size 32:

- gate/up weights are packed as interleaved rows for
  `ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens`;
- down weights are packed for `ggml_turbomind_mul_mat_grouped_total_tokens`;
- activation and output buffers are FP16 because that is the TurboMind ABI and
  V100 Tensor Core execution path.

### TP Contract

For `full_mid=2048` and `tp=8`, each participant owns `mid_shard=256`.

The full reference uses:

```text
gate/up: N = 4096, K = 4096, output [routes, 2048]
down:    N = 4096, K = 2048, output [routes, 4096]
```

The TP8 candidate uses:

```text
gate/up shard: N = 512, K = 4096, output [routes, 256]
down shard:    N = 4096, K = 256, output [routes, 4096]
```

Correctness compares the FP32 sum of eight TP partial down outputs against the
full reference output.

## Definition Of Done

- [x] Sprint plan exists.
- [x] New TP-only TurboMind MXFP4 FFN smoke exists.
- [x] No PP scheduler files are modified.
- [x] CUDA target is added to `Makefile`, including macOS CUDA-required branch.
- [x] Local hygiene passes.
- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [ ] `tokens_per_active=16` and `32` runs pass correctness on all eight V100s.
- [x] Timing output reports full, TP8 compute, TP8 reduce/copy, total, and
  speedups.
- [x] Results are copied to
  `logs/from-cluster/sprint211-tp8-turbomind-ffn/`.
- [x] Sprint 211 document records validation and decision.
- [x] Status/Vision documents are updated.
- [x] Changes are committed with explicit `git add` paths.

## Execution

Implemented `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu` as a separate TP-only
low-bit routed-FFN executable. It uses the public TurboMind C ABI through
`dlopen`, packs deterministic synthetic MXFP4 fixtures, builds a full-width
reference on GPU0, builds eight TP middle shards, runs TurboMind gated-SiLU and
down GEMMs, and reduces the eight partial outputs back to GPU0 for comparison.

During bring-up, the first version used raw device pointer tables and hit an
illegal access. The fix was to match the proven TurboMind tests and pass
16-byte `StridedPtrH` descriptors for packed weights/scales. After that fix,
the tool ran and produced meaningful TP8 low-bit evidence.

## Validation

Build:

```text
make -j80 tools/ds4-v100-tp8-turbomind-ffn-smoke CUDA_ARCH=sm_70
```

Cluster evidence is in `logs/from-cluster/sprint211-tp8-turbomind-ffn/`.

### TP8 MXFP4 Result

Configuration:

```text
experts=6
hidden=4096
full_mid=2048
mid_shard=256
dtype=mxfp4
group_size=32
```

| Tokens/active expert | Routes | Correctness | Full | TP8 compute | TP8 reduce | TP8 total | Compute speedup | Total speedup |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | 96 | FAIL, `nan=378153` | `0.293581 ms` | `0.074752 ms` | `0.485761 ms` | `0.560513 ms` | `3.927x` | `0.524x` |
| 32 | 192 | FAIL, `nan=756305` | `0.347750 ms` | `0.083763 ms` | `0.860789 ms` | `0.944552 ms` | `4.152x` | `0.368x` |
| 64 | 384 | FAIL, `nan=1512469` | `0.603904 ms` | `0.144179 ms` | `1.759132 ms` | `1.903312 ms` | `4.189x` | `0.317x` |

TP8 compute itself is fast, but the current TurboMind MXFP4 path at
`mid_shard=256` produces invalid partials and the simple gather/reduce path
erases the compute win even before addressing correctness.

### TP4 Reference Check

The existing four-GPU TurboMind split test was run as a control to distinguish
generic low-bit TP viability from the TP8 shard-width failure.

| Tokens/active expert | Routes | Correctness | Full | TP4 compute | Copy-inclusive | Compute speedup | Total speedup |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16 | 96 | PASS | `0.2925 ms` | `0.1253 ms` | `0.3761 ms` | `2.333x` | `0.778x` |
| 32 | 192 | PASS | `0.3462 ms` | `0.1342 ms` | `0.5502 ms` | `2.579x` | `0.629x` |
| 64 | 384 | PASS | `0.6019 ms` | `0.1637 ms` | `0.8926 ms` | `3.676x` | `0.674x` |

This means the rejection is specific to the TP8 `mid_shard=256` low-bit path
and/or the simple TP8 output reduction, not to TurboMind MXFP4 TP splitting in
general.

## Decision

Reject TP8 MXFP4 expert execution with the current TurboMind shard shape.

The current best interpretation is:

- TP8 topology and FP16 fixture compute remain viable from Sprints 209-210.
- The real MXFP4 expert path does not currently support a clean TP8
  `mid_shard=256` split.
- TP4 `mid_shard=512` remains correct and shows material compute speedup.
- Simple output gather/reduce is already too expensive at these route counts,
  so future TP work needs either TP4 with a better reduction boundary or a new
  TP8 MXFP4 kernel shape that is explicitly designed for `K/N=256` shards.

Next sprint should pivot to a TP4/PP1 low-bit layer-body path or a new
MXFP4 shard-256 kernel investigation. Do not integrate TP8 into serving.

## Decision Gate

Continue TP8 implementation if:

- TurboMind MXFP4 supports the TP8 shard dimensions;
- TP8 summed partials match full reference within explicit tolerance;
- TP8 compute shows material speedup at 32 active slots or improves clearly at
  the larger optional shape;
- reduce/copy cost does not erase the useful compute signal.

If TurboMind rejects `mid_shard=256`, record that as a real TP8 kernel-layout
blocker and plan either TP4 low-bit experts or a new MXFP4 kernel shape. Do not
work around it by changing the PP scheduler.

## Risks

- Generic TurboMind grouped GEMM may not support the small TP8 shard shape.
- Full-reference and TP8 low-bit paths may differ numerically enough to need a
  tolerance wider than FP16 fixture tests.
- Output reduction through simple peer copies is not a production collective.
- This still does not include attention/KV execution.

## Security

No service exposure. No model weights. Synthetic low-bit fixtures only.

## Dependencies

- Sprint 210 TP8 real-layer fixture.
- Existing TurboMind MXFP4 build on V100.
- V100 build pod or direct node access.
