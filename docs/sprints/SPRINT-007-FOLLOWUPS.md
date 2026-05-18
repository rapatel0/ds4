# SPRINT-007 Follow-Ups

These are not blockers for the Sprint 007 `SHIP` verdict, but they should shape
Sprint 008 and validation hardening.

## Correctness Hardening

- Add a direct parity test against the GGML `block_mxfp4` dequant layout so the
  low-half/high-half nibble ordering cannot regress.
- Add a small official-vector runner around the current `--dump-logprobs`
  diagnostic command so the selected-token comparison is automated instead of
  manually inspected.
- Capture a full-logit local oracle artifact for one short prompt after the
  runtime can emit it without excessive storage or time cost.

## Runtime Guard Hardening

- Decide whether to add the explicit code-level oracle unlock token originally
  sketched in the sprint plan, or keep the narrower CLI diagnostic session gate.
- Add targeted guard tests for source oracle normal generation, source oracle
  diagnostic sessions, MTP rejection, and non-CPU rejection.

## Sprint 008 Inputs

- Use the corrected MXFP4 source layout as the reference for routed expert
  kernel work.
- Keep BF16 as source/converted input only on V100; production dense math should
  target FP16 HMMA with FP32 accumulation rather than broad FP32 GEMMs.
- Treat the Sprint 007 oracle as the first correctness reference for prefill,
  compressed KV, and device-side source-format kernels.
