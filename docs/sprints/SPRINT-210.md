# Sprint 210 - TP8 Real Layer Body Prototype

Date: 2026-05-23
Status: Planned

## Overview

Replace the Sprint 209 synthetic TP8 resident body with a standalone TP-only
layer body that performs real tensor-core GEMM work across all eight V100s.

This sprint should answer whether the TP8 boundary still looks plausible when
useful layer-shaped compute is inside the boundary, not just synthetic scalar
work. It remains a prototype sprint, not serving integration.

## Rationale

Sprint 209 passed the topology gate:

| Shape | Total one-layer latency | Reduce latency | Correctness |
|---:|---:|---:|---|
| 32 tokens | `0.739408 ms` | `0.634680 ms` | ok |
| 64 tokens | `0.876011 ms` | `0.718601 ms` | ok |
| 128 tokens | `1.098461 ms` | `0.840586 ms` | ok |

That proves the separate TP8 boundary and sharded-KV ownership are viable
enough to continue, but it does not prove DS4 layer execution. The next useful
gate is to add real resident tensor-core work:

```text
hidden
  -> TP8 column-parallel gate/up GEMM
  -> gated activation
  -> TP8 row-parallel down GEMM
  -> TP8 hidden reduction
```

The first version may use deterministic FP16 fixtures so that we can measure
shape, residency, reduction, and Tensor Core occupancy without blocking on pack
conversion. The sprint must leave a clear bridge to the low-bit TurboMind
expert body that follows.

## Scope

1. Add a new TP-only CUDA executable for a real layer-body smoke.
2. Use all eight V100s as TP participants.
3. Allocate DS4-shaped KV shards per GPU using the Sprint 209 descriptor math.
4. Allocate resident hidden input/output buffers per GPU.
5. Run a layer-shaped FFN body with real tensor-core GEMMs:
   - column-parallel gate GEMM;
   - column-parallel up GEMM;
   - gated activation;
   - row-parallel down GEMM;
   - TP8 hidden reduction.
6. Time the phases separately:
   - KV allocation/accounting;
   - gate/up GEMM;
   - activation;
   - down GEMM;
   - reduction;
   - total layer body.
7. Run 32, 64, and 128 token shapes on all eight V100s at the 32-slot /
   256K-context planning target.
8. Record the decision for Sprint 211:
   - continue to low-bit TurboMind TP8 experts;
   - continue to sharded attention/KV;
   - or pause TP8 if useful compute does not amortize the boundary.

## Non-Goals

- No generic scheduler implementation.
- No PP scheduler changes.
- No `ds4_v100_scheduler.*` changes.
- No launcher default changes.
- No full-model TP serving.
- No model-weight pack conversion.
- No claim that FP16 fixture compute is the final DS4 precision path.
- No cleanup or commit of unrelated Sprint 207 kernel/runtime work.

## Architecture

New files should stay TP-only:

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp8-real-layer-smoke.cu` | Main TP8 real layer-body executable |
| `logs/from-cluster/sprint210-tp8-real-layer/` | V100 build/run evidence |
| `docs/sprints/SPRINT-210.md` | Sprint plan and outcome |

If helpers are needed, keep them local to the new TP executable first. Do not
create a shared scheduler abstraction. Extraction can happen only after a real
TP runtime direction is proven.

### Compute Contract

The executable should model the FFN side of a DS4 layer as a resident TP8 body:

```text
for each participant gpu p:
  x[p]              : [tokens, hidden] fp16
  gate_w[p]         : [hidden, mid_shard] fp16 fixture
  up_w[p]           : [hidden, mid_shard] fp16 fixture
  down_w[p]         : [mid_shard, hidden] fp16 fixture

  gate[p] = x[p] @ gate_w[p]
  up[p]   = x[p] @ up_w[p]
  mid[p]  = silu(gate[p]) * up[p]
  partial[p] = mid[p] @ down_w[p]

tp8_reduce_sum(partial) -> hidden output on every gpu
```

Use cuBLAS or an existing local GEMM primitive for the first version. The
important point is that the work uses real Tensor Core GEMM shape, not scalar
synthetic loops. Keep weights deterministic and small enough to fit comfortably
with the KV shard allocation.

### Precision Contract

This sprint may use FP16 fixture weights and activations to characterize TP8
shape. It must explicitly document that production DS4 should still move toward
low-bit packed expert weights:

- MXFP4 / FP8 source storage remains the target for experts;
- conversions should happen inside GPU kernels where possible;
- FP16 here is a topology and Tensor Core occupancy fixture, not a final model
  format decision.

## Definition Of Done

- [ ] Sprint plan exists.
- [ ] New TP-only real layer-body executable exists.
- [ ] No PP scheduler files are modified.
- [ ] CUDA target is added to `Makefile`, including macOS CUDA-required branch.
- [ ] Local hygiene passes.
- [ ] V100 build passes with `CUDA_ARCH=sm_70`.
- [ ] 32, 64, and 128 token runs pass correctness on all eight V100s.
- [ ] Timing output separates gate/up GEMM, activation, down GEMM, reduction,
  and total layer time.
- [ ] Results are copied to `logs/from-cluster/sprint210-tp8-real-layer/`.
- [ ] Sprint 210 document records validation and decision.
- [ ] Status/Vision documents are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Decision Gate

Continue TP8 implementation if:

- useful resident GEMM work plus reduction scales better from 32 to 64/128
  tokens than the synthetic boundary alone;
- reduction is no longer the overwhelming majority once realistic GEMM work is
  added;
- correctness passes;
- memory remains compatible with 32-slot/256K planning.

If the real GEMM body is still dominated by reduction at 32/64/128 tokens, stop
TP8 runtime work and return to a monolithic/persistent low-bit routed-FFN
kernel path.

## Risks

- FP16 fixture GEMMs may overstate performance relative to MXFP4/FP8 unpack and
  dequant.
- FP16 fixture GEMMs may understate performance if TurboMind low-bit kernels
  reduce memory traffic materially.
- cuBLAS launch overhead may differ from a fused production executor.
- This does not yet cover sharded attention/KV row selection.

## Security

No service exposure. Do not copy model weights into logs. Cluster logs should
contain only synthetic fixture timings and command output.

## Dependencies

- Sprint 208 TP8 planner/proxy evidence.
- Sprint 209 TP8 sharded-KV and one-layer boundary executable.
- V100 build pod or direct node access.
