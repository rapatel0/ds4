# Sprint 036 8-GPU Gate Rollup

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-036-GATE-CLUSTER-8GPU
```

Result:

- `gate mtp_ffn PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

Focused MTP FFN evidence:

- selected experts matched CPU reference: `0,83,57,141,163,179`
- route weight max abs delta: `2.98023224e-08`
- routed output max abs delta: `7.15255737e-07`
- shared output max abs delta: `1.90734863e-06`
- final FFN output max abs delta: `1.90734863e-06`
- `next_hc` max abs delta: `2.38418579e-06`
- resident MTP FFN arena time: `3.830 ms`

Baseline appliance evidence:

- selected token gate produced expected first token hex `3136`
- replay first generated token id `926`, text `16`, hex `3136`
- replay generated TPS: `0.553769`
- HTTP long request generated TPS: `0.721906`

Remaining readiness blocker:

- `mtp_forward`: MTP raw/SWA attention, logits/top-k, and draft verify/rollback
  are not yet integrated into serving.
