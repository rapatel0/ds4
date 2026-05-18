# SPRINT-008 Follow-Ups

These are not blockers for the Sprint 008 `SHIP` verdict, but they should shape
Sprint 009 and later validation work.

## Correctness Hardening

- Capture a local full-logit source-oracle artifact for one short prompt once
  the runtime can emit it without excessive storage or runtime cost.
- Add a second official-vector case after the source oracle runner can process a
  fixture with a materially different prompt shape.
- Keep `tools/ds4-source-oracle-vector --guard-checks` in the cluster validation
  path for every sprint that touches source-layout session creation.

## Device Source-Format Anchors

- Add an MXFP4 CUDA diagnostic anchor before routed expert production kernels
  consume the packed expert layout.
- Add a BF16-to-F16 or BF16-to-F32 diagnostic anchor for dense BF16 source
  tensors that will be converted for V100 FP16 HMMA paths.
- Consider a bounded F8 row-dot probe after the row-decode anchor is consumed by
  the first dense projection prototype.

## Sprint 009 Inputs

- Use `ds4_v100_kv_budget_for_layer` and the per-stage `kv_*_bytes` fields as
  the admission contract for prompt prefill and compressed KV allocation.
- Use the F8_E4M3_B128 CUDA row-decode anchor as the reference pattern for
  future source-format CUDA probes: explicit view bounds, strided synthetic
  rows, CPU helper parity, and fail-closed malformed inputs.
- Keep normal source-layout generation guarded until V100 prefill/decode is
  validated against the source oracle runner.
