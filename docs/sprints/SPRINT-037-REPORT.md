# Sprint 037 Report: Resident MTP Raw Attention

## Summary

Sprint 037 shipped the resident gpu7 MTP raw/SWA attention rung. The MTP
sidecar arena now feeds `mtp.0.attn_sinks.weight` directly into the CUDA
attention decoder, and the focused smoke proves production raw-KV store plus
128-row ring-cache visibility through wrap positions.

This is still not full `mtp_forward`; the remaining blocker is integrated MTP
attention projection/output, draft logits/top-k, and verify/rollback.

## Code Changes

- Added `ds4_gpu_arena_attention_decode_heads_tensor()` in the CUDA backend.
  It mirrors the existing mmap-backed decode attention path but resolves sinks
  from a `ds4_gpu_arena` F32 view.
- Added the fail-closed CPU stub for the new arena attention API.
- Added `tools/ds4-v100-mtp-attn-smoke`, a focused CUDA smoke using real
  sidecar-resident sinks, synthetic Q/KV, production FP8-plus-F16 raw KV store,
  and CPU sink-aware attention reference math.
- Wired `tools/ds4-v100-mtp-attn-smoke` into `Makefile`.
- Added the `mtp_attn` gate rung to `tools/ds4-v100-gate.sh`, after the
  existing MTP prefix/Q4K/FFN rungs and before the remaining `mtp_forward`
  readiness blocker.

## Validation

Local:

```bash
make tools/ds4-v100-mtp-attn-smoke.o ds4_v100_mtp.o ds4_gpu_arena_stub.o ds4_cpu.o
bash -n tools/ds4-v100-gate.sh
git diff --check
```

Cluster focused smoke:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-attn-smoke
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 ./tools/ds4-v100-mtp-attn-smoke \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-037-MTP-ATTN/mtp_attn.report
```

Focused result:

- positions checked: `0,1,127,128,129`
- raw-cache wrap check at `pos=129`: `raw_start=2`, `current_row=1`,
  `current_visible=1`, `oldest_row=2`, `oldest_visible=1`
- global attention max abs delta: `1.27183739e-08`
- `mtp_attn_smoke PASS`

Full cluster gate:

- `gate mtp_attn PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

Replay evidence:

- first generated token id: `926`
- first generated token text: `16`
- first generated token hex: `3136`
- replay generated TPS: `0.551384`

Artifacts:

- `docs/sprints/drafts/SPRINT-037-MTP-ATTN/mtp_attn.report`
- `docs/sprints/drafts/SPRINT-037-GATE-CLUSTER-8GPU/mtp_attn.report`
- `docs/sprints/drafts/SPRINT-037-GATE-CLUSTER-8GPU/ROLLUP.md`

## Remaining Work

The next sprint should advance from isolated raw attention semantics to a
complete one-token MTP block:

- MTP attention projection chain from `mtp_input_hc`.
- Grouped Q8_0 attention output projection and HC expansion.
- MTP output norm/logits/top-k parity.
- Replay-level draft verify/rollback before speculative serving is enabled.
