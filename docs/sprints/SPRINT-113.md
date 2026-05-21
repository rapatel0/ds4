# Sprint 113 - Direct FFN Delta Accumulation

Date: 2026-05-20

## Objective

Reduce repeated host/API and per-slot device work in the fused TurboMind
appliance path by making the batched FFN path write directly into the final FFN
delta tensor.

## Context

Sprint 112 confirmed that tiny scalar F8 kernel tweaks are not enough. The
fused appliance profile still shows large API-side waiting, and the current FFN
batch path has a clean avoidable host boundary:

- `execute_ffn_delta_batch()` owns `input_ptrs_t` in persistent batch scratch.
- In HC batch serving, `ffn_inputs[slot]` is the stable scratch view
  `batch_scratch->ffn_norm[slot]`.
- The TurboMind batch wrapper currently rebuilds a host vector of those pointers
  and uploads it on every layer call.
- The shared F8 batch path already accepts an existing pointer table.

This sprint should split "build/upload the pointer table" from "consume the
pointer table" for TurboMind, cache the stable FFN table in
`ds4_v100_layer_batch_scratch`, and add an opt-in direct-delta path:

1. shared F8 FFN writes into contiguous `ffn_delta_batch`;
2. TurboMind routed expert scatter atomically accumulates into that same tensor;
3. the current routed-output plus per-slot add loop is skipped.

## Plan

1. Add contiguous FFN norm and delta batch tensors to layer-batch scratch.
2. Add TurboMind routed FFN batch APIs that consume an already-populated device
   pointer table:
   - separate gate/up;
   - fused gate_up.
3. Add TurboMind routed FFN APIs that accumulate into an existing output tensor
   instead of clearing it first.
4. Add scratch metadata for cached FFN input pointer tables:
   - valid flag;
   - slot count;
   - minimum row byte span.
5. In `execute_ffn_delta_batch()`, use the cached pointer table only when the
   incoming `ffn_inputs[]` are exactly the scratch `ffn_norm[]` views.
6. Add guarded `DS4_V100_FFN_DIRECT_DELTA=1` execution:
   - shared F8 path writes directly to `ffn_delta_batch`;
   - TurboMind routed path accumulates into `ffn_delta_batch`;
   - fallback remains the current routed/shared/add sequence.
7. Keep the existing upload-wrapper path as fallback for non-scratch callers and
   focused tests.
8. Validate correctness and measure the production 8-slot/256K fused appliance.

## Definition of Done

- [x] V100 `sm_70` build passes for replay and scheduler/token smoke tests.
- [x] Full scheduler smoke passes with the fused appliance.
- [x] Selected-token smoke passes with expected hex `3136`.
- [x] 8-slot/256K same-binary A/B is recorded against Sprint112 default with
      `DS4_V100_FFN_DIRECT_DELTA=0/1`.
- [x] Decision is documented:
  - default direct delta only if throughput improves or is neutral
    with a clear API reduction;
  - otherwise keep fallback behavior and document the result.

## Implementation

- `ds4_v100_layer_batch_scratch` now owns contiguous `ffn_norm_batch` and
  `ffn_delta_batch` tensors, with the existing per-slot `ffn_norm[]` and
  `ffn_delta[]` entries represented as stable views.
- TurboMind routed FFN wrappers now have ptr-table entry points that can consume
  an already-populated device pointer table instead of rebuilding and uploading
  one inside every call.
- TurboMind routed FFN wrappers also have accumulate entry points that skip the
  routed-output clear and atomically add into an existing output tensor.
- `execute_ffn_delta_batch()` keeps the old fallback path, and enables the new
  path only with `DS4_V100_FFN_DIRECT_DELTA=1` when the batch uses persistent
  scratch, batched shared F8, and scratch-owned FFN norm/delta views.
- The launcher, env example, and k8s config expose
  `DS4_V100_FFN_DIRECT_DELTA`, defaulting to `0`.

## Validation

Cluster build:

```text
CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-replay \
  tests/cuda_source_dtypes_smoke \
  tests/cuda_v100_projection_attention_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke
```

Direct-delta correctness run:

```text
DS4_V100_FFN_DIRECT_DELTA=1
DS4_V100_BATCH_SHARED_F8=1
DS4_V100_TURBOMIND_FUSED_GATE_UP=1
DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1
DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=0
DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0
```

Results:

- `cuda_source_dtypes_smoke`: passed.
- `cuda_v100_projection_attention_smoke`: passed.
- `cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 --slots 8`:
  passed, `tm_layers=43`.
- `cuda_v100_selected_token_smoke --expected-token-hex 3136`: passed, selected
  token id `926`, logit `35.254894`.

## Throughput

Same-binary 8-slot/256K A/B, fused Sprint111 appliance,
`tokens=16`, `requests=8`, `active_microbatch=8`:

| Config | Generated tok/s | Continuation tok/s | Token match | Decision |
|---|---:|---:|---:|---|
| `DS4_V100_FFN_DIRECT_DELTA=0` | `33.589285` | `31.489955` | 8/8 | keep default |
| `DS4_V100_FFN_DIRECT_DELTA=1` | `33.360404` | `31.275379` | 8/8 | opt-in only |

Artifact directory:

```text
logs/from-cluster/sprint113-direct-delta/
```

The candidate is correct but slightly slower on the primary target, so this
sprint keeps `DS4_V100_FFN_DIRECT_DELTA=0` as the production default.

The pre-sprint TurboMind total-tokens ABI recheck also stayed rejected:

| Config | Generated tok/s | Continuation tok/s | Token match |
|---|---:|---:|---:|
| `DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1` | `33.555903` | `31.458659` | 8/8 |
| `DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=0` | `30.370739` | `28.472568` | 8/8 |

## Risks

- The pointer table is only reusable when the input tensor view identities are
  stable. Any non-scratch caller must keep the old upload path.
- This removes a small host upload, not the larger TurboMind row-count wait
  rejected again after Sprint112. Throughput gain may be modest.
- If future code reallocates batch scratch, cached metadata must be reset with
  the scratch allocation.
