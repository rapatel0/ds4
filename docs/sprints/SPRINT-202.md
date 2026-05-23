# Sprint 202 - TP4 Routed-FFN Compute Envelope

Date: 2026-05-23
Status: Completed

## Objective

Measure the compute side of the TP4/EP decision using real TurboMind MXFP4
routed-FFN kernels across four V100s.

Sprint 201 measured the full-layer TP4 communication boundary. Sprint 202
answers the matching compute question: if the routed expert intermediate
dimension is split four ways and the four shards run concurrently, is the
compute speedup large enough to justify the TP4 boundary in a future full-layer
runtime?

## Rationale

Do not expand the old routed-only TP2 overlay. It moves hidden state in and out
of one FFN boundary and already regressed served throughput.

This sprint is different: it does not wire a routed-only production overlay.
It measures the TP4 routed-expert compute envelope that would live inside a
broader full-layer TP4/EP topology where dense and routed work remain inside the
same TP boundary.

## Scope

1. Add a four-GPU TurboMind TP split benchmark.
2. Split DS4 routed FFN `mid=2048` into four `512`-wide shards.
3. Use the real copied TurboMind MXFP4 grouped gated-SiLU and down kernels.
4. Compare full one-GPU output against the sum of four TP partial outputs.
5. Report:
   - full one-GPU FFN time;
   - per-shard time;
   - concurrent compute time;
   - compute speedup;
   - conservative total-with-copy time;
   - total-with-copy speedup;
   - input/output payload size.
6. Build and run on the V100 pod for practical route shapes.

## Non-Goals

- No production scheduler integration.
- No claim that TP4 is ready for serving from this benchmark alone.
- No NCCL dependency.
- No dense attention/shared-layer TP implementation in this sprint.

## Implementation Map

| File | Work |
|---|---|
| `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp` | New four-GPU TP4 routed-FFN compute and payload proxy. |
| `kernels/turbomind/ggml-turbomind/CMakeLists.txt` | Add the new benchmark target. |
| `logs/from-cluster/sprint202-tp4-routed-ffn/` | Store V100 build/run evidence. |
| `TEMP_STATUS_REPORT_016.md` | Record current results and decision. |
| `docs/sprints/VISION.md`, `STATUS.md`, `EXPERIMENT-STATUS.md` | Update TP4 direction. |

## Definition Of Done

- [x] Sprint plan exists.
- [x] `test_ggml_turbomind_tp_split_4gpu` builds on V100.
- [x] Correctness passes for at least one practical route shape.
- [x] Benchmark reports compute and total-with-copy speedups.
- [x] Benchmark warmup lifecycle bug is fixed and validated.
- [x] Evidence is copied into `logs/from-cluster/sprint202-tp4-routed-ffn/`.
- [x] Status, vision, experiment, and TEMP report documents are updated.
- [x] Changes are committed.

## Decision Gate

If four-way TP compute speedup is weak or erased by conservative payload copies,
do not pursue a production TP4 routed-expert path until dense/full-layer TP is
implemented.

If four-way TP compute speedup is strong and the copy-inclusive result is still
competitive, the next sprint should implement a bounded full-layer TP4/EP slice
over a small layer span rather than another proxy.

## Implementation

Added `test_ggml_turbomind_tp_split_4gpu`, a four-GPU extension of the existing
two-GPU TP split benchmark. The test:

- builds synthetic finite MXFP4 gate/up/down expert fixtures;
- packs a full one-GPU routed FFN and four `512`-wide TP shards;
- runs the real TurboMind grouped gated-SiLU and down kernels;
- compares the full output against the sum of four partial outputs;
- reports compute-only and conservative copy-inclusive timing.

During validation, the first combined run exposed a benchmark lifecycle bug:
the warmup launched the full one-GPU reference on GPU0 and then shard 0 on GPU0
in a different stream before the full stream completed. Both calls share the
TurboMind per-device workspace, so the overlap could wedge GPU0 while peer GPUs
sat idle. The benchmark now separates full-reference warmup from TP-shard
warmup and synchronizes the full stream before shard warmup begins.

## Validation

V100 build passed:

```text
cmake --build build/turbomind-v100 --target test_ggml_turbomind_tp_split_4gpu -j80
```

Measured on devices `0,1,2,3` after the warmup fix:

| Tokens/active expert | Total routes | Full 1-GPU | TP4 concurrent compute | Compute speedup | Total with copies | Copy-inclusive speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---|
| `1` | `6` | `0.1457 ms` | `0.0543 ms` | `2.686x` | `0.1479 ms` | `0.986x` | PASS |
| `16` | `96` | `0.2920 ms` | `0.1242 ms` | `2.350x` | `0.3729 ms` | `0.783x` | PASS |
| `128` | `768` | `1.1553 ms` | `0.3178 ms` | `3.636x` | `1.6936 ms` | `0.682x` | PASS |

Evidence:

```text
logs/from-cluster/sprint202-tp4-routed-ffn/
```

Key logs:

```text
tp4-split-debug-tpa1-warmup1.log
tp4-split-debug-tpa1-warmup1-fixed.log
tp4-split-0-1-2-3-cases-1-16-128-fixed.log
```

## Decision

The TP4 routed-expert compute envelope is strong, but a routed-only copy-in /
copy-out overlay is still the wrong architecture.

At `96-768` routes, compute-only TP4 gives `2.35x-3.64x` speedup. That is enough
to make TP4 worth pursuing inside a full-layer topology. However, conservative
full-hidden input/output copies erase the win at those same practical route
counts: `0.78x` and `0.68x` copy-inclusive speedups.

This confirms the Sprint 201 interpretation. The next implementation should be
a bounded full-layer TP4/EP slice that keeps hidden state resident across
attention, shared dense work, routed experts, and the necessary collectives. Do
not build another routed-only overlay.
