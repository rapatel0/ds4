# Sprint 038 Cluster Gate Rollup

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-038-GATE-CLUSTER-8GPU
```

## Result

- `gate mtp_attn PASS`
- `gate scheduler_output_head PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

## MTP Attention Evidence

- Raw/cache wrap:
  `global_max_abs=1.27183739e-08`
- Integrated attention:
  - `q_heads max_abs=2.14576721e-06`
  - `kv_row max_abs=0.000867605209`
  - `heads max_abs=2.14576721e-06`
  - `attn_out max_abs=0.258209229`
  - `next_hc max_abs=0.19461441`

## Base Appliance Evidence

- Replay first generated token id `926`, hex `3136`.
- HTTP long smoke request 1 first token id `926`, hex `3136`,
  continuation `148.485 ms`.
- HTTP long smoke request 2 first token id `926`, hex `3136`,
  continuation `144.103 ms`.

## Post-run GPU State

All 8 V100s returned to `0MiB / 32768MiB`; no GPU processes remained.
