# Sprint 069 Report: Appliance Launcher Soak Harness

## Outcome

`SHIP`.

Sprint 069 adds a reusable launcher soak harness and runs the practical
4-slot, 1M-context appliance profile through `tools/ds4-v100-run-appliance.sh`.
This validates the operator path rather than only the replay binary or
benchmark harness.

## Implementation

- Added `tools/ds4-v100-appliance-soak.sh`.
- The harness starts `tools/ds4-v100-run-appliance.sh` with explicit
  `DS4_V100_*` environment values.
- It validates:
  - `/health`;
  - `/v100/status`;
  - `/metrics`;
  - concurrent `/v100/selected-token` responses;
  - expected first-token hex;
  - presence of `timing_ms.async_pipeline`.
- It archives:
  - request JSON;
  - health/status/metrics;
  - raw responses;
  - startup env;
  - resolved command;
  - server log;
  - GPU utilization samples;
  - summary JSON.

## Validation

Local:

- `chmod +x tools/ds4-v100-appliance-soak.sh`
- `bash -n tools/ds4-v100-appliance-soak.sh`
- `tools/ds4-v100-appliance-soak.sh --help`
- `git diff --check`

V100:

- `bash -n tools/ds4-v100-appliance-soak.sh`
- Practical launcher soak:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1800 \
  tools/ds4-v100-appliance-soak.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --slots 4 \
  --active-microbatch 4 \
  --queue-policy sequential \
  --tokens 16 \
  --requests 4 \
  --expected-token-hex 3136 \
  --async-pipeline-mode auto \
  --port 18420 \
  --sample-ms 500 \
  --log-dir logs/sprint069-appliance-soak
```

Evidence:

- `logs/from-cluster/sprint069-appliance-soak/summary.json`
- `logs/from-cluster/sprint069-appliance-soak/responses.json`
- `logs/from-cluster/sprint069-appliance-soak/status_before.json`
- `logs/from-cluster/sprint069-appliance-soak/status_after.json`
- `logs/from-cluster/sprint069-appliance-soak/metrics_before.txt`
- `logs/from-cluster/sprint069-appliance-soak/metrics_after.txt`
- `logs/from-cluster/sprint069-appliance-soak/runtime/startup.env`
- `logs/from-cluster/sprint069-appliance-soak/runtime/command.txt`
- `logs/from-cluster/sprint069-appliance-soak/server.log`
- `logs/from-cluster/sprint069-appliance-soak/gpu_util.csv`

## Result

Summary:

| Metric | Value |
|---|---:|
| Requests | `4` |
| HTTP 200 | `4` |
| Token matches | `4` |
| Errors | `0` |
| Generated tokens | `64` |
| Continuation tokens | `60` |
| Aggregate generated tok/s | `7.518610` |
| Aggregate continuation tok/s | `7.048697` |
| Average latency | `8510.724 ms` |
| Async mode | `per-step` |

The run is slightly below the Sprint 067 direct benchmark's 1M/4-slot
`8.617368` generated tok/s, but it includes the launcher path and broader
status/metrics collection. More importantly, it proves the deployment path now
reaches the same class of practical serving behavior without manual replay
flags.

## Decision

- Keep `tools/ds4-v100-appliance-soak.sh` as the deployment smoke/soak harness.
- Use the Sprint 069 4-slot, 1M-context result as the current practical
  launched-appliance baseline.
- The next throughput work should target a real speed lever: MTP draft commit
  or stream/event inter-stage handoff. Additional launcher plumbing alone will
  not move tok/s materially.
