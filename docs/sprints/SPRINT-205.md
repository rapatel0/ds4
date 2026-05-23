# Sprint 205 - Async Root Resident TP4 Reduction Gate

Date: 2026-05-23
Status: Completed

## Objective

Add and validate a concurrent root gather/reduce/broadcast path for the resident
TP4 layer-slice benchmark, then decide whether small-payload 96-route decode
can beat the async doubling path from Sprint 204.

## Rationale

Sprint 204 proved that concurrent peer movement matters, but its pairwise
doubling path still missed the production 96-route gate on repeat. For small
decode payloads, root can be cheaper than doubling because it performs one
reduction kernel on GPU0 instead of local add kernels on every participant.

Sprint 203's root was synchronous. This sprint tests the missing variant:
asynchronous peer gather to root, root reduction, and asynchronous broadcast
back to peers.

## Scope

1. Add `DS4_TP4_RESIDENT_ALGO=root_async`.
2. Keep the same resident TP4 layer-slice benchmark and correctness gate.
3. Measure 96-route and 768-route shapes over 43 layers.
4. Compare against Sprint 203 root and Sprint 204 doubling_async.

## Non-Goals

- No production scheduler integration.
- No NCCL dependency.
- No custom fused collective kernel beyond root-side sum.

## Definition Of Done

- [x] Sprint plan exists.
- [x] `root_async` builds in `test_ggml_turbomind_tp4_resident_layer_slice`.
- [x] V100 correctness passes at 96 and 768 routes.
- [x] Benchmark reports whether root_async beats root and doubling_async.
- [x] Evidence is copied into
      `logs/from-cluster/sprint205-tp4-root-async/`.
- [x] Status, vision, experiment, and TEMP report documents are updated.
- [x] Changes are committed.

## Decision Gate

If `root_async` does not reliably clear the 96-route 43-layer gate, pause TP4
production decode work and pivot to persistent fused routed-FFN implementation.

If it does clear the gate, keep TP4 alive for one more sprint with a production
collective-quality gate before scheduler integration.

## Implementation

Added `DS4_TP4_RESIDENT_ALGO=root_async` to
`test_ggml_turbomind_tp4_resident_layer_slice`. The path:

1. asynchronously gathers peer partial hidden outputs from GPUs 1-3 into GPU0;
2. synchronizes those peer streams;
3. runs one root-side `sum4_half_kernel`;
4. asynchronously broadcasts the reduced hidden state back to peers.

The first attempt used cross-device CUDA events and failed with
`invalid resource handle`; the final implementation uses stream
synchronization after the async peer copies and passes correctness.

## Validation

V100 build passed:

```text
cmake --build build/turbomind-v100 --target test_ggml_turbomind_tp4_resident_layer_slice -j80
```

Measured on devices `0,1,2,3`:

| Shape | Algo | Full ref | Resident TP4 | Speedup | Boundary/iter | Correctness |
|---|---|---:|---:|---:|---:|---|
| `96 routes x 4 layers` | root_async | `1.2683 ms` | `1.3071 ms` | `0.970x` | `18.00 MiB` | PASS |
| `768 routes x 4 layers` | root_async | `4.7405 ms` | `5.4744 ms` | `0.866x` | `144.00 MiB` | PASS |
| `96 routes x 43 layers` | root_async | `11.8966 ms` | `13.8286 ms` | `0.860x` | `193.50 MiB` | PASS |

Evidence:

```text
logs/from-cluster/sprint205-tp4-root-async/
```

## Decision

`root_async` is rejected.

It is correct, but it is slower than the one-GPU reference and slower than
Sprint 204's `doubling_async` path. The production 96-route decode shape is
`0.860x` over 43 layers, so this does not justify TP4 scheduler integration.

This closes the current TP4 decode branch. TP4 remains plausible for
larger-batch/prefill work, but the next high-throughput practical-serving
sprint should pivot to the persistent fused routed-FFN executor.
