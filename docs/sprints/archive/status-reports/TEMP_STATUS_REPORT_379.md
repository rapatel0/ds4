# TEMP Status Report 379: Fused Gated-SiLU Gate

Date: 2026-05-25

## Current Focus

Sprint 379 is testing `--fused-gated-silu-gate`, the next Vision throughput
gate after compact MoE.

The intended target is the routed-FFN activation boundary:

```text
MXFP4 gate/up GEMM -> fp16 gate_up -> DS4 clamp/SwiGLU kernel -> fp16 gated
```

The candidate tries to use TurboMind's fused gated-SiLU epilogue directly:

```text
MXFP4 gate/up GEMM + gated-SiLU epilogue -> fp16 gated
```

## Implemented

- Added `--fused-gated-silu-gate`.
- Added `DS4_V100_TP_EP_FUSED_GATED_SILU=1`.
- Added `tools/ds4-v100-tp-ep-profile.py --fused-gated-silu`.
- Added `tools/ds4-v100-tp-ep-profile.py --routed-ffn-norm-input` so the
  actual standalone-clamp branch can be A/B tested.
- Added a narrow TurboMind ABI:
  `ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens`.
- Added TurboMind epilogue support for DS4's clamp semantics:
  `gate = min(gate, 10)`, `up = clamp(up, -10, 10)`,
  `out = silu(gate) * up`.
- Added scaffold metadata:
  - `fused_gated_silu_gate`
  - `routed_ffn_norm_input_gate`
  - `routed_gate_standalone_swiglu`

The gate defaults off and local validation passed:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100 build passed:

```text
cmake --build build/turbomind-v100 --target ggml-turbomind -j80
nm -D build/turbomind-v100/libggml-turbomind.so | grep ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Direct A/B Evidence

Artifact root:

```text
/workspace/logs/sprint379-fused-gated-silu
```

### Current Serving-Shaped Branch

Shape:

```text
direct-token-major
32 slots
256K context
position 262080
1 generated token
model-router routes
compact-MoE decode
routed-FFN normalized input off
```

| Mode | First token | Decode tok/s | EP ms | Standalone SwiGLU |
|---|---:|---:|---:|---:|
| control | 54639 | 68.313313 | 16.873766 | 0 |
| fused flag | 54639 | 67.795224 | 17.999301 | 0 |

Finding: the current serving-shaped branch already has no standalone
clamp/SwiGLU launch. The fused flag is parity-clean here, but it is effectively
a no-op for the S-E target.

### Routed-Normalized Branch

Shape:

```text
direct-token-major
32 slots
256K context
position 262080
1 generated token
model-router routes
compact-MoE decode
routed-FFN normalized input on
```

| Mode | First token | Decode tok/s | EP ms | Standalone SwiGLU |
|---|---:|---:|---:|---:|
| clamped control | 41432 | 45.368432 | 34.189967 | 1 |
| generic fused epilogue | 54639 | 57.367413 | 27.038617 | 0 |

Finding: the generic TurboMind gated-SiLU epilogue removes the standalone
launch and improves the routed-normalized direct proxy, but it changes the
first token. It is not DS4-equivalent to the clamped path.

### DS4-Clamped ABI Evidence

The DS4-clamped ABI exports and builds. A narrow layer-0 EP-only V100 run at
the same slot/context/router shape exercises the fused clamped gate itself and
passes:

```text
32 slots
256K context
position 262080
layer 0
model-router routes
routed-FFN normalized input on
compact-MoE decode
```

| Mode | Worst gate ms | Worst down ms | Worst EP ms | Result |
|---|---:|---:|---:|---|
| two-step clamped gate | 4.102144 | 0.223232 | 4.254720 | PASS |
| fused DS4-clamped gate | 0.622592 | 0.181248 | 0.803840 | PASS |

That is a useful isolated result: the clamped TurboMind epilogue launches and
removes the expensive standalone clamped SwiGLU boundary for this layer-0
EP-only shape.

However, the resident serving-shaped direct A/B is not valid yet. With
`--routed-ffn-norm-input --fused-gated-silu`, the `32` slot / `256K` /
`position=262080` resident direct run fails at layer 0 before the routed gate
is reached:

```text
returncode: 4
tp_ep_token_major_item step 0 layer 0 rc 4 FAIL
scaffold_pass_invocations: 0
```

The failure comes from the resident `ds4_v100_tp_runtime_dense_kv_slice`
precheck (`kv_result.max_abs != 0.0`) before `run_decode_loop()` calls
`run_gate_selected()`. A same-binary routed-normalized control rerun passes
with first token `41432`, `59.381346` direct generated tok/s, and
`routed_gate_standalone_swiglu=1`. The current serving-shaped fused flag
without routed-normalized input also passes with first token `54639` and
`68.824485` direct generated tok/s.

## Decision

Do not promote `--fused-gated-silu-gate`.

The generic epilogue is rejected for correctness because it changes the routed
normalized first token. The DS4-clamped epilogue is promising in isolation but
is not serving-validated because the resident direct A/B fails before the gate
executes. Keep the ABI and flag as default-off diagnostics only.

## Next

Next work should either:

- diagnose the resident dense-KV precheck failure under
  `routed-normalized + fused-gated-silu`, then rerun the serving A/B; or
- add a deterministic gate-output parity harness that compares the fused
  DS4-clamped ABI against the two-step clamped reference before returning to
  serving.
