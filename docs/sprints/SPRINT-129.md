# Sprint 129 - TurboMind Dispatch Policy Probe

Date: 2026-05-21

## Objective

Check whether TurboMind's non-default grouped GEMM dispatch policies can improve
the current Sprint 128 compact routed-expert path before moving to a larger
persistent routed-FFN implementation. The wrapper previously hardcoded
`DispatchPolicy::kDefault` for all TurboMind C ABI calls.

## Implementation

- Added `DS4_V100_TURBOMIND_DISPATCH_POLICY` to the TurboMind C ABI wrapper.
- Supported safe policies:
  - `default`: existing heuristic/cache behavior.
  - `reuse`: request cache lookup before heuristic fallback.
- Kept `measure` and `append` available only behind
  `DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1`.
- Hardened the launcher so `measure` and `append` fail config validation unless
  the unsafe override is explicit.
- Recorded the dispatch policy and unsafe override in launcher `--check`,
  startup env logs, deployment env example, and the appliance runbook.

## Validation

Local/static:

```text
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
DS4_V100_TURBOMIND_DISPATCH_POLICY=measure \
  tools/ds4-v100-run-appliance.sh --check --allow-missing

rc=1
DS4_V100_TURBOMIND_DISPATCH_POLICY=measure requires
DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1

DS4_V100_TURBOMIND_DISPATCH_POLICY=reuse \
  tools/ds4-v100-run-appliance.sh --check --allow-missing

turbomind_dispatch_policy=reuse
turbomind_allow_unsafe_measure=0
```

Cluster build on `llamacpp-build-8gpu` under `/workspace/ds4`:

```text
cmake --build build/turbomind-v100-s127 \
  --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80
```

Focused TurboMind gate/up test:

| Policy | Result |
|---|---|
| `default` | PASS |
| `reuse` | PASS |
| `measure` with unsafe override before hardening | PASS in focused test, but not representative |

Full scheduler:

```text
DS4_V100_TURBOMIND_LIB=/workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so \
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 \
DS4_V100_TURBOMIND_DISPATCH_POLICY=default \
./tests/cuda_v100_full_scheduler_smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/ds4-appliance-full-tm-fused-s111 \
  --slots 16 --ctx 262144 --expect-tm-layers 43

ok: layers=43 tm_layers=43

DS4_V100_TURBOMIND_DISPATCH_POLICY=reuse ...

ok: layers=43 tm_layers=43
```

Unsafe `measure` full scheduler result before the guard:

```text
[TM][FATAL] measurer.cu:83 Check failed: status == cudaSuccess invalid argument
exit code 134
```

## Results

Served A/B at `ctx=262144`, `slots=16`, `active_microbatch=16`,
`tokens=16`, `requests=16`, `warmup_requests=1`, existing Sprint 111 fused
appliance plus Sprint 128 compact schedule:

| Policy | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|
| `default` | `45.840691` | `42.975648` | 16/16 token match |
| `reuse` | `45.813841` | `42.950476` | 16/16 token match |

The safe alternate policy is effectively neutral. The measured policy is not a
production candidate because the full appliance path can abort inside
TurboMind's measurer.

## Decision

Keep `DS4_V100_TURBOMIND_DISPATCH_POLICY=default` as the production default.
Do not promote `reuse`; it did not improve served throughput. Keep `measure`
and `append` blocked unless `DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1` is set
for a focused diagnostic.

The next sprint should move to the larger implementation path: a narrow
DS4-only opt-in persistent routed-FFN branch in `cuda_tm_routed_mxfp4_packed_impl()`
that targets the current best shape first rather than more generic TurboMind
dispatch tuning.

## Risk

The new policy knob is intentionally conservative. Directly setting
`DS4_V100_TURBOMIND_DISPATCH_POLICY=measure` outside the launcher is also
guarded in the wrapper unless `DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1` is
present, because the full scheduler already proved that path can terminate the
process.
