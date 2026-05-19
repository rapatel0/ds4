# Sprint 036 Report: Resident MTP FFN Slice

## Summary

Sprint 036 shipped a resident gpu7 MTP FFN slice. The MTP sidecar arena now
feeds HC FFN control, FFN RMS norm, bias-router selection, Q4_K routed experts,
Q8_0 shared experts, routed+shared accumulation, and HC expansion to `next_hc`.

This is not full `mtp_forward`; MTP attention/raw cache, logits/top-k, and draft
verify/rollback remain the readiness blocker.

## Code Changes

- Added `ds4_v100_mtp_sidecar_f32_matrix_view()` for resident 2D F32 sidecar
  tensors such as `mtp.0.hc_ffn_fn.weight` and `mtp.0.ffn_gate_inp.weight`.
- Added `ds4_gpu_arena_router_select_bias_tensor()` for resident MTP bias
  top-k routing without falling back through `cuda_model_range_ptr()`.
- Added `ds4_gpu_arena_hc_split_weighted_sum_tensor()` for resident MTP HC
  scale/base control bytes.
- Added `tools/ds4-v100-mtp-ffn-smoke`, a focused CUDA smoke that compares the
  resident MTP FFN slice against a CPU reference built from the same sidecar
  bytes.
- Wired `mtp_ffn` into `tools/ds4-v100-gate.sh` after `mtp_q4k` and before the
  remaining `mtp_forward` readiness blocker.

## Validation

Local:

```bash
make tools/ds4-v100-mtp-ffn-smoke.o ds4_v100_mtp.o ds4_gpu_arena_stub.o ds4_cpu.o
bash -n tools/ds4-v100-gate.sh
git diff --check
```

Cluster focused smoke:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-ffn-smoke
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 ./tools/ds4-v100-mtp-ffn-smoke \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-036-MTP-FFN/mtp_ffn.report
```

Focused result:

- selected experts: `0,83,57,141,163,179`
- route weight max abs delta: `2.98023224e-08`
- routed output max abs delta: `7.15255737e-07`
- shared output max abs delta: `1.90734863e-06`
- final FFN output max abs delta: `1.90734863e-06`
- `next_hc` max abs delta: `2.38418579e-06`
- `mtp_ffn_smoke PASS`

Full cluster gate:

- `gate mtp_ffn PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

Artifacts:

- `docs/sprints/drafts/SPRINT-036-MTP-FFN/mtp_ffn.report`
- `docs/sprints/drafts/SPRINT-036-GATE-CLUSTER-8GPU/ROLLUP.md`

## Remaining Work

The next sprint should implement the rest of `mtp_forward`:

- MTP raw/SWA attention cache update and attention output.
- MTP output norm/logits/top-k parity against a trusted reference.
- Draft verify/rollback semantics before enabling speculative serving.
