# Sprint 037 Cluster Gate Rollup

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-037-GATE-CLUSTER-8GPU
```

## Result

- `gate build PASS`
- `gate mtp_attn PASS`
- `gate scheduler_output_head PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

## Focused MTP Attention Evidence

From `mtp_attn.report`:

- positions checked: `0,1,127,128,129`
- wrap mapping: `pos=129 raw_start=2 current_row=1 current_visible=1 oldest_row=2 oldest_visible=1`
- global max abs delta: `1.27183739e-08`
- `mtp_attn_smoke PASS`

## Replay Evidence

From `v100_replay_tool.log`:

- first generated token id: `926`
- first generated token text: `16`
- first generated token hex: `3136`
- generated tokens: `2`
- generated TPS: `0.551384`

## HTTP Evidence

From `v100_appliance_http_long.log`:

- request 1 first token id: `926`
- request 1 first token hex: `3136`
- request 1 continuation ms: `150.779`
- request 2 first token id: `926`
- request 2 first token hex: `3136`
- request 2 continuation ms: `144.825`
- health/status: `ok`
