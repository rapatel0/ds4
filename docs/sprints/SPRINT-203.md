# Sprint 203 - Resident TP4 Layer-Slice Gate

Date: 2026-05-23
Status: Completed

## Objective

Build and validate a bounded resident TP4 layer-slice benchmark that combines
the real TurboMind MXFP4 routed-FFN split with the hidden-state layer boundary
inside one four-GPU device-resident loop.

## Rationale

Sprint 201 measured TP4 layer-boundary traffic by itself. Sprint 202 measured
TP4 routed-expert compute by itself and found a benchmark lifecycle bug in the
process. The corrected result was clear: TP4 expert compute scales, but
routed-only copy-in/copy-out erases the win.

The next useful gate is therefore not another routed-only overlay. We need to
measure the shape the runtime would actually want: hidden state resident on a
four-GPU TP island, TP-sharded routed FFN work, and a layer-boundary reduction
that produces the next resident hidden state.

## Scope

1. Add a V100-buildable resident TP4 layer-slice benchmark.
2. Reuse the Sprint 202 synthetic finite MXFP4 expert fixtures and real
   TurboMind grouped gated-SiLU/down kernels.
3. Split DS4 routed FFN `mid=2048` into four `512`-wide shards.
4. For each measured layer:
   - run all four TP shards;
   - reduce the four partial hidden outputs into the next hidden state;
   - keep the next hidden state resident for the next layer.
5. Compare the resident TP4 result against a one-GPU full routed-FFN reference
   over the same layer count.
6. Report full-reference time, resident TP4 time, per-layer time, speedup, and
   boundary payload.
7. Build and run on the V100 pod for the important route shapes.

## Non-Goals

- No production scheduler integration yet.
- No DS4 attention, norms, residuals, KV, or router implementation in this
  benchmark.
- No claim that this is a final TP4 production collective.
- No 8-GPU TP8 or TP4/PP2 implementation in this sprint.

## Implementation Map

| File | Work |
|---|---|
| `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp` | Wrap `main()` so shared TP4 fixture/packing helpers can be included by the resident slice benchmark. |
| `kernels/turbomind/ggml-turbomind/test_tp4_resident_layer_slice.cu` | New resident TP4 layer-slice benchmark. |
| `kernels/turbomind/ggml-turbomind/CMakeLists.txt` | Add `test_ggml_turbomind_tp4_resident_layer_slice`. |
| `logs/from-cluster/sprint203-tp4-resident-layer-slice/` | Store V100 build/run evidence. |
| `TEMP_STATUS_REPORT_017.md` | Record current results and decision. |
| `docs/sprints/VISION.md`, `STATUS.md`, `EXPERIMENT-STATUS.md` | Update practical-serving direction. |

## Definition Of Done

- [x] Sprint plan exists.
- [x] `test_ggml_turbomind_tp4_resident_layer_slice` builds on V100.
- [x] At least one route shape passes resident TP4 correctness against the
      one-GPU reference.
- [x] Benchmark reports full-reference and resident TP4 per-layer timing.
- [x] Benchmark records the resident boundary payload per measured iteration.
- [x] Evidence is copied into
      `logs/from-cluster/sprint203-tp4-resident-layer-slice/`.
- [x] Status, vision, experiment, and TEMP report documents are updated.
- [x] Changes are committed.

## Decision Gate

If the resident TP4 slice is still slower at the 96-route 16-slot/256K shape,
but improves at 768 routes, TP4 should be treated as a high-batch/prefill path
until a better collective or denser serving shape exists.

If the resident slice is competitive at 96 routes, the next sprint should move
from benchmark to a bounded runtime slice over a tiny layer span.

If correctness fails, stop TP4 production work and fix the layout/accumulation
contract before any scheduler integration.

## Implementation

Added `test_ggml_turbomind_tp4_resident_layer_slice`, a CUDA benchmark that
includes the Sprint 202 TP4 split helpers, packs the same finite MXFP4
gate/up/down fixtures, and runs a resident per-layer loop:

1. run four TP4 routed-FFN shards with real TurboMind grouped gated-SiLU and
   down kernels;
2. reduce partial hidden outputs into the next hidden state;
3. keep that hidden state resident for the next layer;
4. compare the result against a one-GPU full-width reference over the same
   layer count.

The benchmark supports:

```text
DS4_TP_SPLIT4_GPUS=0,1,2,3
DS4_TP_SPLIT_CASES=1,16,128
DS4_TP4_RESIDENT_LAYERS=1..43
DS4_TP4_RESIDENT_ALGO=root|doubling
DS4_TP4_RESIDENT_WARMUP_ITERS=N
DS4_TP4_RESIDENT_BENCH_ITERS=N
```

## Validation

V100 build passed:

```text
cmake --build build/turbomind-v100 --target test_ggml_turbomind_tp4_resident_layer_slice -j80
```

Measured on devices `0,1,2,3`:

| Shape | Algo | Full ref | Resident TP4 | Speedup | Boundary/iter | Correctness |
|---|---|---:|---:|---:|---:|---|
| `6 routes x 1 layer` | root | `0.1591 ms` | `0.1726 ms` | `0.921x` | `0.28 MiB` | PASS |
| `96 routes x 4 layers` | root | `1.1625 ms` | `1.4949 ms` | `0.778x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | root | `4.3276 ms` | `7.0362 ms` | `0.615x` | `144.00 MiB` | PASS |
| `96 routes x 4 layers` | doubling | `1.2657 ms` | `2.1035 ms` | `0.602x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | doubling | `4.7441 ms` | `9.4932 ms` | `0.500x` | `144.00 MiB` | PASS |
| `96 routes x 43 layers` | root | `12.9784 ms` | `15.7303 ms` | `0.825x` | `193.50 MiB` | PASS |
| `768 routes x 43 layers` | root | `44.0628 ms` | `74.8701 ms` | `0.589x` | `1548.00 MiB` | PASS |

Evidence:

```text
logs/from-cluster/sprint203-tp4-resident-layer-slice/
```

## Decision

Do not move this root/doubling resident TP4 boundary into the production
scheduler.

The resident TP4 slice is correct and uses the real MXFP4 expert kernels, but
the naive reduction boundary still loses to a one-GPU full-width routed-FFN
reference at both the production 96-route shape and the larger 768-route shape.
The simple hand-rolled doubling variant is slower than root because it performs
sequential peer exchanges and per-device add kernels rather than a concurrent
ring/tree collective.

This narrows the next TP work: before scheduler integration, TP4 needs a real
collective implementation or fused reduction boundary. Without that, the project
should pivot back to the other serious lever: a true persistent/fused routed-FFN
executor that reduces global handoffs inside one GPU.
