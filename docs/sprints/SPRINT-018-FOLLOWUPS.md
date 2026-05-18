# SPRINT-018 Follow-Ups

These are not blockers for the Sprint 018 `SHIP` verdict, but they still block
a deployable DS4 V100 appliance.

## Full Attention Softmax And Compressed-KV Output

- **What:** Use the descriptor-bound q/kv projection surfaces to run real
  attention softmax over raw and compressed KV rows, then produce a semantic
  attention output instead of the Sprint 018 bounded output-projection proxy.
- **Why:** Sprint 018 proves real projection/residual/norm surfaces but not full
  attention semantics.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 019.
- **Files:** `ds4_v100_layer_state.*`, `ds4_cuda.cu`, attention/layer smokes.

## Combined Attention Plus FFN Layer Slice

- **What:** Compose descriptor-bound attention output with the existing
  router-selected FFN path through one scheduler-owned layer state.
- **Why:** Serving needs a coherent next hidden state, not isolated attention
  and FFN surfaces.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 019+.
- **Files:** layer runtime/scheduler, CUDA smokes, gate script.

## Real-Model Selected-Token Gate

- **What:** Drive a bounded real-model descriptor-bound path to output-head
  logits and selected-token comparison.
- **Why:** The gate still cannot validate a real selected token from the
  layer-scheduled path.
- **Severity:** Critical.
- **Suggested sprint:** Sprint 020+.
- **Files:** scheduler/layer runtime, output-head path, gate script.

## Production Arena Reuse

- **What:** Execute layer-state smokes against resident stage arenas instead of
  partial test arenas.
- **Why:** Sprint 018 still duplicates source weights into a bounded arena for
  validation.
- **Severity:** Important.
- **Suggested sprint:** Sprint 019+.
- **Files:** V100 context/residency wiring, scheduler runtime.

## Summary

| Item | Severity | Suggested Sprint | Files |
|---|---|---|---|
| Full attention softmax/compressed-KV output | Critical | Sprint 019 | layer state/CUDA/tests |
| Combined attention plus FFN layer slice | Critical | Sprint 019+ | scheduler/tests/gate |
| Real-model selected-token gate | Critical | Sprint 020+ | scheduler/output/gate |
| Production arena reuse | Important | Sprint 019+ | context/residency/runtime |
