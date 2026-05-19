# Sprint 035 Gate Rollup

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-035-GATE-CLUSTER-8GPU
```

## Result

```text
gate mtp_q4k PASS
gate v100_replay_tool PASS
gate v100_appliance_http PASS
gate v100_appliance_http_long PASS
gate readiness NOT_READY missing=mtp_forward
gate summary PASS failures=0 ready=false
```

## MTP Q4_K Evidence

```text
mtp_q4k_tensor gate dtype=q4_k experts=256 rows=2048 cols=4096 row_stride=2304 expert_stride=4718592 bytes=1207959552
mtp_q4k_tensor up dtype=q4_k experts=256 rows=2048 cols=4096 row_stride=2304 expert_stride=4718592 bytes=1207959552
mtp_q4k_tensor down dtype=q4_k experts=256 rows=4096 cols=2048 row_stride=1152 expert_stride=4718592 bytes=1207959552
mtp_q4k_routed arena_ms=3.476 reference_ms=38.194 max_abs=1.43051147e-06 tol=0.05 PASS
```

## Replay Evidence

```text
first token hex=3136
prompt_replay=3432.057 ms
continuation_decode=144.740 ms
generated_tokens_per_second=0.557913
```
