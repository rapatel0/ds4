# Sprint 053 Report: Continuous Token-Step Microbatching

## Result

`SHIP`.

## Changes Implemented

1. Added a reusable same-length batch generation primitive:
   - `ds4_v100_replay_generate_batch`;
   - `ds4_v100_replay_generate_first_token_batch` remains as a wrapper.
2. Updated `tools/ds4-v100-replay` serving:
   - non-MTP pending requests with the same `tokens` value now run through the
     multi-token batch API;
   - one-request, mixed-token-count, and MTP requests keep the serial fallback;
   - `/v100/status` reports `tensor_batched_slots=true` when
     `active_microbatch > 1`;
   - status and metrics now include `tensor_batched_groups`,
     `tensor_batched_requests`, and `tensor_batched_tokens`.
3. Updated `tools/ds4-v100-sustained-decode-bench.sh`:
   - captures `server_status_before.json` and `server_status_after.json`;
   - embeds those status snapshots in each case `result.json`;
   - uses Python stdlib HTTP for status capture so the pod does not need
     `curl`.
4. Updated the runbook and vision with the shipped behavior and cluster result.

## Validation

Local:

```bash
cc -fsyntax-only -I. ds4_v100_replay.c
cc -fsyntax-only -I. tools/ds4-v100-replay.c
make ds4_v100_replay.o tools/ds4-v100-replay.o
bash -n tools/ds4-v100-sustained-decode-bench.sh
```

Cluster build:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_ARCH=sm_70 make tools/ds4-v100-replay &&
  bash -n tools/ds4-v100-sustained-decode-bench.sh
'
```

## Cluster Execution

Executed on `llamacpp-build-8gpu` (`gpu-01`) with all eight V100s visible:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint053 &&
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  timeout 1200 bash ./tools/ds4-v100-sustained-decode-bench.sh \
    --model /models/DSv4-Flash-256e-fixed.gguf \
    --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
    --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
    --ctx-tiers 1048576 \
    --slot-tiers 1,2 \
    --queue-policies sequential \
    --tokens 16 \
    --requests 4 \
    --warmup-requests 1 \
    --expected-token-hex 3136 \
    --sample-ms 500 \
    --log-dir logs/sprint053-token-step-batching
'
```

Result:

| ctx | slots | status | token match | generated tok/s | continuation tok/s | avg GPU util | max GPU util | tensor batch evidence |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1048576 | 1 | PASS | 4/4 | 3.291466 | 3.085749 | 10.768% | 22.000% | 0 groups / 0 requests / 0 tokens |
| 1048576 | 2 | PASS | 4/4 | 3.371659 | 3.160931 | 11.133% | 22.000% | 1 group / 2 requests / 32 tokens |

Artifacts:

- `logs/from-cluster/sprint053-token-step-batching/sustained_decode.tsv`
- `logs/from-cluster/sprint053-token-step-batching/sustained_decode.json`
- `logs/from-cluster/sprint053-token-step-batching/cases/case_1_ctx1048576_s1_sequential_tok16/result.json`
- `logs/from-cluster/sprint053-token-step-batching/cases/case_2_ctx1048576_s2_sequential_tok16/result.json`
- per-case `server_status_before.json`, `server_status_after.json`,
  `server.log`, and `gpu_util.csv`

## Assessment

The serving path now truly executes a same-length two-request batch, and the
status counters prove it. Performance improved only slightly: generated tok/s
rose from `3.291466` to `3.371659`, about `2.4%`, and average GPU utilization
stayed near `11%`.

That is a useful result because it narrows the problem. The next performance
work should not spend more time on basic request-loop plumbing. The remaining
bottleneck is hot-path occupancy: low-bit expert kernels in the real decode
path, routed expert batching, fewer small launches, and eventually persistent
MoE scheduling.
