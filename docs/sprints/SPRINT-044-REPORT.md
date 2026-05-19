# Sprint 044 Report: Throughput Optimization And Operating Envelope

## Result

`SHIP`.

Sprint 044 closed the `throughput_optimization` readiness blocker by adding a
before/after startup benchmark and shipping the first real optimization:
parallel stage open/upload in `ds4_v100_replay_open`. The served base path still
returns the expected first token bytes `3136`, and the full 8-GPU gate now
passes with:

```text
gate	throughput_optimization	PASS
gate	production_deployment	PASS
gate	readiness	NOT_READY	missing=mtp_speculative_serving
gate	summary	PASS	failures=0 ready=false
```

## Implementation Summary

- Added threaded stage open/upload for the eight `ds4_v100_stage_scheduler`
  instances inside `ds4_v100_replay_open`.
- Kept tokenizer open, model mmap, and model-fd registration serial before
  stage workers start.
- Added `serial_open` as an explicit replay option and CLI `--serial-open`
  fallback.
- Added `tools/ds4-v100-replay --open-only` for startup/upload timing without
  prompt replay or generation.
- Added `tools/ds4-v100-throughput-bench.sh`, which runs serial open-only,
  parallel open-only, and a normal two-token correctness replay.
- Wired the benchmark into `tools/ds4-v100-gate.sh` as
  `throughput_optimization`.
- Updated the appliance runbook with the startup benchmark command and artifact
  contract.

## Local Validation

```bash
bash -n tools/ds4-v100-throughput-bench.sh tools/ds4-v100-gate.sh
```

```bash
cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. \
  -D_FILE_OFFSET_BITS=64 \
  -c -o /tmp/ds4_v100_replay.o ds4_v100_replay.c
```

```bash
cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. \
  -D_FILE_OFFSET_BITS=64 \
  -c -o /tmp/ds4-v100-replay.o tools/ds4-v100-replay.c
```

```bash
./tools/ds4-v100-throughput-bench.sh --help
```

## Cluster Build And CLI Check

```bash
cd /workspace/ds4-sprint044
CUDA_ARCH=sm_70 make tools/ds4-v100-replay
./tools/ds4-v100-replay --help | grep -E -- '--open-only|--serial-open'
```

The CUDA build passed and the generated help documents both new flags.

## Focused Throughput Benchmark

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1500 \
  ./tools/ds4-v100-throughput-bench.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --expected-token-hex 3136 \
  --min-speedup 1.05 \
  --log-dir docs/sprints/drafts/SPRINT-044-THROUGHPUT
```

Result:

```text
serial_open_ms	343989.990
parallel_open_ms	63032.135
speedup	5.457375
replay_open_ms	70272.460
prompt_replay_ms	4379.105
continuation_decode_ms	152.225
continuation_tokens_per_second	6.569219
first_token_hex	3136
verdict	PASS
```

Artifact:

- `docs/sprints/drafts/SPRINT-044-THROUGHPUT/`

## Full Gate

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 4200 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-044-GATE-CLUSTER-8GPU
```

Result:

```text
gate	mtp_verify	PASS
gate	v100_replay_tool	PASS
gate	throughput_optimization	PASS
gate	v100_appliance_http	PASS
gate	v100_appliance_http_long	PASS
gate	production_deployment	PASS
gate	readiness	NOT_READY	missing=mtp_speculative_serving
gate	summary	PASS	failures=0 ready=false
```

Full-gate throughput benchmark:

```text
serial_open_ms	227418.984
parallel_open_ms	59449.323
speedup	3.825426
replay_open_ms	63892.546
prompt_replay_ms	3429.294
continuation_decode_ms	144.105
continuation_tokens_per_second	6.939385
first_token_hex	3136
verdict	PASS
```

The full-gate `v100_replay_tool` artifact also shows the optimized default
open/upload path:

```text
open_total_ms	63202.989
prompt_replay_ms	4105.007
continuation_decode_ms	143.823
first_token_hex	3136
uploaded_bytes	156142862684
```

Production deployment smoke remained green with optimized open/upload:

```text
open_total_ms	68515.592
prompt_replay_ms	3553.735
continuation_decode_ms	142.750
first_token_hex	3136
```

Artifact:

- `docs/sprints/drafts/SPRINT-044-GATE-CLUSTER-8GPU/`

## Remaining Blocker

The next readiness blocker is `mtp_speculative_serving`: the MTP sidecar is
resident and its one-token verify/rollback path is gated, but the HTTP appliance
still serves only the base one-slot path with `mtp_enabled=false`.

Multi-slot aggregate throughput and broader context-tier benchmarking remain
future work. Sprint044 improves cold startup/upload and records one-slot decode
timing; it does not claim multi-slot scheduling or aggregate slot throughput.
