# Sprint 039 Gate Rollup

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-039-GATE-CLUSTER-8GPU
```

## Result

- `gate mtp_logits PASS`
- `gate scheduler_output_head PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

## Focused MTP Logits Evidence

- Top-1: CPU `65615`, GPU `65615`.
- Top-5 token parity: exact.
- Max selected-logit absolute delta: `9.53674316e-07`.
- Base output-head upload: `1,059,061,760` bytes.
- Free after MTP sidecar plus output-head upload: `28,878,307,328` bytes.
- Reserve requirement: `4,294,967,296` bytes.

## Post-Run GPU State

All 8 V100s returned to `0 MiB / 32768 MiB` after the gate.
