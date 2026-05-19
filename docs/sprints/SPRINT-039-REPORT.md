# Sprint 039 Report: Resident MTP Logits and Top-K Parity

## Summary

Sprint 039 added the resident MTP logits/top-k proof. The V100 path now runs
the MTP-specific HC-head collapse from sidecar-resident `mtp.0.hc_head_*`
tensors, applies sidecar-resident `mtp.0.norm.weight`, projects through the
base model BF16 `output.weight`, reads logits, and selects top-k candidates.

The focused smoke and full 8-GPU gate both pass. Readiness remains correctly
blocked on `missing=mtp_forward` because full one-token MTP block composition
and draft verify/rollback semantics are not implemented yet.

## Implementation

- Added `ds4_gpu_arena_output_hc_weights_tensor` for resident-arena F32
  scale/base HC-head sigmoid weights.
- Added `tools/ds4-v100-mtp-logits-smoke.c`.
- Wired `tools/ds4-v100-mtp-logits-smoke` into `Makefile`.
- Wired `mtp_logits` into `tools/ds4-v100-gate.sh` when both `--mtp-model` and
  `--pack-index` are supplied.
- Kept readiness conservative: `mtp_logits` passing advances the gate rung,
  but the gate still reports `missing=mtp_forward`.

## Validation

Local:

```bash
make tools/ds4-v100-mtp-logits-smoke.o ds4_v100_mtp.o ds4_v100_context.o ds4_gpu_arena_stub.o ds4_cpu.o ds4_source_formats.o
bash -n tools/ds4-v100-gate.sh
```

Cluster focused build:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-logits-smoke
```

Cluster focused smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 \
  ./tools/ds4-v100-mtp-logits-smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-039-MTP-LOGITS/mtp_logits.report
```

Focused result:

- `cpu_top1=65615`
- `gpu_top1=65615`
- `max_abs=9.53674316e-07`
- `PASS`

Top-k parity:

| Rank | Token | CPU Logit | GPU Logit | Delta |
|---:|---:|---:|---:|---:|
| 1 | 65615 | 18.6035843 | 18.6035843 | 0 |
| 2 | 8764 | 16.8168888 | 16.8168888 | 0 |
| 3 | 5865 | 16.3385696 | 16.3385696 | 0 |
| 4 | 2630 | 16.2020359 | 16.2020359 | 0 |
| 5 | 41163 | 15.8609848 | 15.8609858 | 9.53674316e-07 |

Memory evidence:

- MTP sidecar arena: `3,807,601,408` bytes.
- MTP sidecar upload: `3,807,600,108` bytes.
- MTP sidecar post-upload free: `29,937,369,088` bytes.
- Base output-head upload: `1,059,061,760` bytes.
- Post-output-upload free: `28,878,307,328` bytes.
- Reserve requirement: `4,294,967,296` bytes.

Cluster full gate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU
```

Full gate result:

- `gate mtp_logits PASS`
- `gate scheduler_output_head PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

Post-run GPU state:

- All 8 V100s returned to `0 MiB / 32768 MiB`.
- No running GPU processes remained.

## Artifacts

- `docs/sprints/drafts/SPRINT-039-MTP-LOGITS/mtp_logits.report`
- `docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU/mtp_logits.report`
- `docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU/mtp_logits.log`
- `docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU/v100_replay_tool.log`
- `docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU/v100_appliance_http.log`
- `docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU/v100_appliance_http_long.log`

## Remaining Blockers

- Full one-token resident MTP block composition from prefix through attention,
  FFN, output norm, logits, and selected draft token.
- Draft verify/rollback semantics against target-model KV and slot state.
- Throughput benchmarking after MTP correctness is wired into the appliance
  loop.
