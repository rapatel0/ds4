# Sprint 131 - TurboMind Indexed-A Routed Activation Probe

Date: 2026-05-21

## Objective

Test a low-risk fusion-adjacent routed-FFN optimization before starting a
deeper custom software-pipelined expert mainloop: avoid materializing
route-expanded FP16 activations for TurboMind gate/up GEMMs by using
TurboMind's indexed-A support.

## Implementation

Added guarded `DS4_V100_TURBOMIND_INDEXED_A=1` support in the packed
TurboMind routed path.

When enabled:

- route build writes `sorted_tokens[row] = sorted_pairs[row] / n_routes`;
- the activation cast stores one FP16 row per active token instead of one row
  per active route;
- gate/up TurboMind grouped GEMMs receive the compact activation buffer plus
  `token_indices`;
- down GEMM remains route-expanded because its input is the routed middle
  activation buffer.

This keeps the existing packed MXFP4 expert weights, compact active-expert
schedule, fused gate/up path, and output scatter semantics unchanged. The
launcher and appliance env sample expose the flag, defaulting to `0`.

## Validation

Local launcher sanity:

```text
bash -n tools/ds4-v100-run-appliance.sh
DS4_V100_TURBOMIND_INDEXED_A=1 tools/ds4-v100-run-appliance.sh --check --allow-missing
```

V100 build:

```text
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4 && make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke CUDA_ARCH=sm_70 -j80'
```

Full 43-layer smoke, both settings:

```text
DS4_V100_TURBOMIND_INDEXED_A=0 ... cuda_v100_full_scheduler_smoke --slots 16
DS4_V100_TURBOMIND_INDEXED_A=1 ... cuda_v100_full_scheduler_smoke --slots 16
```

Both runs passed:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 uploaded_bytes=156142896212 expert_last=26 ok
```

Served A/B on the current fused compact appliance:

```text
appliance: /workspace/ds4-appliance-full-tm-fused-s111
lib:       /workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so
ctx:       262144
slots:     16
tokens:    16
requests:  16
```

| Mode | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---|
| compact fused control | `45.663281` | `42.809326` | 16/16 token match |
| compact fused + indexed-A | `45.789937` | `42.928066` | 16/16 token match |

## Decision

Keep `DS4_V100_TURBOMIND_INDEXED_A=0` by default.

The path is correct and reduces route-expanded activation scratch, but the
served gain is inside run noise. This confirms the main point from Sprint 130:
the worthwhile fusion/software-pipeline target is inside the packed expert
GEMM mainloop and epilogue, not another wrapper-level data movement tweak.
