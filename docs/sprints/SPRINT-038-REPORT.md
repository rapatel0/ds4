# Sprint 038 Report: Resident MTP Integrated Attention Slice

## Summary

Sprint 038 strengthened the `mtp_attn` gate from raw/SWA attention only to an
integrated resident MTP attention slice on gpu7. The smoke now composes real
sidecar-resident HC attention control, attention norm, Q/KV projections, Q/KV
norms, production FP8 raw-cache store, sink-aware attention decode, grouped
Q8_0 attention output, and HC expansion back to `[4 x 4096]`.

The focused smoke and the full 8-GPU appliance gate both pass. Readiness remains
correctly blocked on `missing=mtp_forward` because MTP logits/top-k and
draft verify/rollback are not implemented yet.

## Implementation

- Extended `tools/ds4-v100-mtp-attn-smoke.c`.
- Added CPU sidecar-byte reference helpers for the integrated attention slice:
  F32 matmul, RMSNorm, HC split/expand, Q8_0 matmul, FP8 KV rounding, and
  sink-aware attention.
- Kept raw attention/cache-wrap tolerance tight through `--max-abs-tol`
  default `0.002`.
- Added `--integrated-max-abs-tol` default `0.5` for the grouped Q8_0 output
  projection and expanded HC comparison. The larger tolerance is isolated to
  CPU serial accumulation versus V100 warp-reduction Q8 paths; raw attention
  and Q/KV/head checks remain much tighter.

## Validation

Local:

```bash
make tools/ds4-v100-mtp-attn-smoke.o ds4_v100_mtp.o ds4_gpu_arena_stub.o ds4_cpu.o
bash -n tools/ds4-v100-gate.sh
git diff --check
```

Cluster focused build:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-attn-smoke
```

Cluster focused smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 \
  ./tools/ds4-v100-mtp-attn-smoke \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-038-MTP-ATTN/mtp_attn.report
```

Focused result:

- `mtp_attn_raw_smoke PASS`
- `q_heads max_abs=2.14576721e-06`
- `kv_row max_abs=0.000867605209`
- `heads max_abs=2.14576721e-06`
- `attn_out max_abs=0.258209229`
- `next_hc max_abs=0.19461441`
- `mtp_attn_integrated_summary ... PASS`
- `mtp_attn_smoke PASS`

After removing unused device-reference allocations, the final focused rebuild
and smoke also passed:

- `CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-attn-smoke`
- `mtp_attn_smoke PASS`

Cluster full gate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-038-GATE-CLUSTER-8GPU
```

Full gate result:

- `gate mtp_attn PASS`
- `gate scheduler_output_head PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

Post-run GPU state:

- All 8 V100s returned to `0MiB / 32768MiB`.
- No running GPU processes remained.

## Artifacts

- `docs/sprints/drafts/SPRINT-038-MTP-ATTN/mtp_attn.report`
- `docs/sprints/drafts/SPRINT-038-MTP-ATTN/mtp_attn_final.report`
- `docs/sprints/drafts/SPRINT-038-GATE-CLUSTER-8GPU/mtp_attn.report`
- `docs/sprints/drafts/SPRINT-038-GATE-CLUSTER-8GPU/v100_replay_tool.log`
- `docs/sprints/drafts/SPRINT-038-GATE-CLUSTER-8GPU/v100_appliance_http.log`
- `docs/sprints/drafts/SPRINT-038-GATE-CLUSTER-8GPU/v100_appliance_http_long.log`

## Remaining Blockers

- MTP output norm/logits/top-k parity.
- One-token resident MTP block composition from prefix through attention and
  FFN into draft logits.
- Draft verify/rollback semantics against target-model state.
