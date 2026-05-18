# SPRINT-017 Follow-Ups

These are not blockers for the Sprint 017 `SHIP` verdict, but they still block
a deployable DS4 V100 appliance.

## Attention, Residual, Norm, And HC Layer Slice

- **What:** Extend `ds4_v100_layer_state` execution beyond router/FFN into
  descriptor-bound attention projections, compressed KV/indexer updates,
  RMSNorm/control tensors, residual adds, and HC transforms.
- **Why:** Sprint 017 created the scheduler-owned descriptor surface, but it
  still cannot produce a next hidden state.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 018.
- **Files:** `ds4_v100_layer_state.*`, `ds4_cuda.cu`, CUDA smokes, gate script.

## Real-Model Selected-Token Gate

- **What:** Drive a bounded real-model descriptor-bound path to output-head
  logits and selected-token comparison.
- **Why:** The selected-token evidence still comes from bounded/synthetic
  smokes, not a real layer-scheduled model path.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 019+.
- **Files:** scheduler/layer runtime, output-head path, `tools/ds4-v100-gate.sh`.

## Production Arena Reuse

- **What:** Let layer-state execution consume the resident stage arena instead
  of the partial smoke arena sized by `ds4_v100_layer_state_ffn_arena_span`.
- **Why:** The state API reports the right span, but production must avoid
  duplicating resident source weights.
- **Severity:** Important.
- **Suggested sprint:** Sprint 018+.
- **Files:** V100 context/residency wiring, scheduler runtime.

## Bias-Router Layer Coverage

- **What:** Add representative state and router execution coverage for layers
  that use `exp_probs_b` instead of `ffn_gate_tid2eid`.
- **Why:** Sprint 017 validates layer 2, which is a hash-router layer.
- **Severity:** Important.
- **Suggested sprint:** After attention/layer slice or when broad layer walking
  begins.
- **Files:** `ds4_v100_layer_state.*`, router tests, gate script.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Attention/residual/norm/HC layer slice | Critical | Sprint 018 | layer state/CUDA/gate |
| Real-model selected-token gate | Critical | Sprint 019+ | scheduler/output/gate |
| Production arena reuse | Important | Sprint 018+ | context/residency/runtime |
| Bias-router layer coverage | Important | After layer slice | layer state/router tests |
