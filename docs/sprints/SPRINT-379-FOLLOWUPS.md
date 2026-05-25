# Sprint 379 Follow-Ups

## Resident Dense-KV Precheck Failure Under Fused Clamped Gate

- **What**: Diagnose why the resident direct serving harness fails at layer 0
  with rc `4` before `run_gate_selected()` executes when both
  `--routed-ffn-norm-input-gate` and `--fused-gated-silu-gate` are enabled.
- **Why**: Sprint 379's DS4-clamped TurboMind ABI passes an EP-only layer-0
  V100 run and is much faster than the two-step clamped gate, but the
  serving-shaped A/B cannot complete because
  `ds4_v100_tp_runtime_dense_kv_slice()` reports non-zero `max_abs` in the
  resident precheck. Same-binary routed-normalized control and serving-shaped
  fused-without-routed-normalized runs pass.
- **Severity**: Important. This blocks any serving promotion decision for the
  fused clamped gate.
- **Suggested sprint**: Next time S-E is revisited; otherwise defer behind
  S-F TP-sharded expert A/B.
- **Files**:
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`,
  `ds4_v100_tp_runtime.cu`,
  `tools/ds4-v100-tp-ep-profile.py`.

## Deterministic Fused-Gate Parity Harness

- **What**: Add a narrow V100 parity harness that compares
  `ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens` output
  directly against the existing `mmgt + routed_fused_gate_up_swiglu_clamp`
  reference for the same routes, weights, and activations.
- **Why**: The EP-only timing proves the clamped ABI launches and is fast, but
  it does not prove output parity. The resident serving A/B currently fails
  before the gate, so a direct gate-output comparison would isolate kernel
  correctness from the rest of the serving harness.
- **Severity**: Important.
- **Suggested sprint**: Same sprint as the dense-KV diagnosis if S-E is
  revisited.
- **Files**:
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`,
  `kernels/turbomind/ggml-turbomind/api.cc`,
  `kernels/turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h`.

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| Resident dense-KV precheck failure under fused clamped gate | Important | Revisit S-E or after S-F | `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, `ds4_v100_tp_runtime.cu`, `tools/ds4-v100-tp-ep-profile.py` |
| Deterministic fused-gate parity harness | Important | Revisit S-E or after S-F | `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, `kernels/turbomind/ggml-turbomind/api.cc`, `kernels/turbomind/lmdeploy/src/turbomind/kernels/gemm/epilogue.h` |
