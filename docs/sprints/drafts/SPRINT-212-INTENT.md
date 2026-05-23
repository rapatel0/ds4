# Sprint 212 Intent - TP4/PP1 Low-Bit Layer Body Pivot

Date: 2026-05-23

## Seed Prompt

Continue the high-throughput practical-serving vision after Sprint 211 rejected
the current TP8 MXFP4 shard shape. Stay in separate TP-only files. Do not add a
generic scheduler and do not modify `ds4_v100_scheduler.*`.

## Orientation Summary

- Sprint 209 proved TP8 topology and sharded KV ownership.
- Sprint 210 proved useful resident FP16 GEMM work inside the TP8 boundary.
- Sprint 211 proved the current TurboMind MXFP4 `mid_shard=256` TP8 expert
  shape is not numerically valid, while the existing TP4 `mid_shard=512`
  TurboMind control remains correct.
- The evidence now points to TP4/PP1 low-bit layer ownership as the next
  practical branch, or a future custom TP8 shard-256 kernel.
- This sprint should implement a TP4 layer-body smoke in `tools/`, not rely only
  on the existing TurboMind test binary.

## Relevant Code Areas

- `tools/ds4-v100-tp8-turbomind-ffn-smoke.cu`: Sprint 211 low-bit TP8 tool;
  has fixture packing and root reduction logic but TP8 correctness fails.
- `kernels/turbomind/ggml-turbomind/test_tp_split_4gpu.cpp`: known-correct
  TP4 MXFP4 split reference.
- `tools/ds4-v100-tp8-real-layer-smoke.cu`: phase timing / TP topology style.
- `Makefile`: new TP-only CUDA target.

## Constraints

- New TP-only implementation files.
- No PP/layer scheduler modifications.
- No launcher defaults.
- Synthetic MXFP4 fixtures only; no model weights in logs.
- V100 validation must use four GPUs in one NVLink island first.

## Success Criteria

- Add a TP4 low-bit layer-body executable under `tools/`.
- Use TurboMind MXFP4 gate/up and down for full reference and four TP shards.
- Verify TP4 partial-sum correctness at practical route shapes.
- Include a resident reduction timing path, not only copy-inclusive timing.
- Compare against Sprint 211 TP8 failure and the existing TP4 control.
- Decide whether TP4/PP1 should become the next real TP runtime branch.

## Verification Strategy

- Local hygiene and macOS CUDA guard.
- V100 build with `CUDA_ARCH=sm_70`.
- Run `tokens_per_active=16`, `32`, and `64` on GPUs `0,1,2,3`.
- Store logs under `logs/from-cluster/sprint212-tp4-lowbit-layer/`.
- Update sprint/status/vision docs and commit explicit files.

## Uncertainty

- Correctness: Low. Existing TP4 TurboMind split passes.
- Scope: Medium. The new tool should be a layer-body smoke, not a scheduler.
- Architecture: Low. Separate TP-only files remain the rule.

## Open Questions

- Does a resident device-side TP4 reduction improve materially over the
  existing copy-inclusive TP4 result?
- Are route counts near 192/384 enough to amortize TP4 reduction?
- Does TP4/PP1 memory planning still preserve 32-slot/256K headroom once MTP
  and output ownership are included?

## Vision Context

Sprint 212 is the topology pivot after TP8 low-bit rejection. It should preserve
the TP-only branch and produce evidence for or against TP4/PP1 as the next
practical-serving implementation path.
