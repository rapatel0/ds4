# DS4 V100 Appliance Runbook

This runbook covers the current production deployment package for the DS4 V100
appliance on the 8x 32 GiB V100 host. The default served endpoint is the
verified one-slot base model path. An explicit MTP verify mode can also expose
the gated one-token MTP draft/verify diagnostics in the same resident HTTP
process.

## Scope

Supported today:

- Source-layout DSv4 Flash base model.
- 8x V100 layer-sharded resident runtime.
- One active slot.
- Sequential loopback HTTP requests.
- `/health`, `/status`, `/v100/status`, `/metrics`.
- `POST /v100/selected-token`.
- Up to 64 generated tokens per request.
- Optional MTP verify diagnostics with `DS4_V100_MTP_SERVING=verify`.
- Operator launcher, env file, systemd template, Kubernetes template, and
  deployment smoke.

Not supported today:

- Multi-token MTP draft commit without recomputing the base target token.
- Multi-slot scheduling.
- Concurrent HTTP requests.
- Streaming responses.
- OpenAI-compatible API.
- External unauthenticated exposure.

## Build

On the V100 build pod or host:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-replay
```

For the full readiness gate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-045-GATE-CLUSTER-8GPU
```

## Required Files

The current cluster convention is:

```text
/models/DSv4-Flash-256e-fixed.gguf
/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv
```

The served base appliance needs the source model and pack index. The MTP model
is required when `DS4_V100_MTP_SERVING=verify` and by the full readiness gate.

## Config

Start from:

```text
deploy/v100/ds4-v100-appliance.env.example
```

Important fields:

```text
DS4_V100_MODEL=/models/DSv4-Flash-256e-fixed.gguf
DS4_V100_MTP_MODEL=/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_V100_PACK_INDEX=docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv
DS4_V100_CTX=1048576
DS4_V100_SLOTS=1
DS4_V100_HOST=127.0.0.1
DS4_V100_PORT=18080
DS4_V100_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DS4_V100_REQUIRE_GPUS=8
DS4_V100_RESERVE_MIB=4096
DS4_V100_SERVE_MODE=base
DS4_V100_MTP_SERVING=off
DS4_V100_MTP_TOP_K=5
DS4_V100_MTP_GPU=7
```

Validate a config without starting the service:

```bash
./tools/ds4-v100-run-appliance.sh \
  --env deploy/v100/ds4-v100-appliance.env.example \
  --check \
  --allow-missing
```

On the V100 host, omit `--allow-missing`; the launcher must see model files,
eight visible GPUs, and at least the configured reserve before it starts.

## Start

Interactive start:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-run-appliance.sh \
  --env deploy/v100/ds4-v100-appliance.env.example
```

The launcher prints the resolved command to
`$DS4_V100_LOG_DIR/command.txt`, writes the resolved startup config to
`$DS4_V100_LOG_DIR/startup.env`, and then execs:

```bash
./tools/ds4-v100-replay --serve ...
```

Use `DS4_V100_MAX_REQUESTS=N` for bounded smoke runs. Use `0` for a supervised
operator session.

## Supervision

Systemd template:

```text
deploy/v100/ds4-v100-appliance.service
```

Expected install shape:

```bash
sudo mkdir -p /etc/ds4-v100
sudo cp deploy/v100/ds4-v100-appliance.env.example /etc/ds4-v100/appliance.env
sudo cp deploy/v100/ds4-v100-appliance.service /etc/systemd/system/ds4-v100-appliance.service
sudo systemctl daemon-reload
sudo systemctl start ds4-v100-appliance
```

Kubernetes template:

```text
deploy/v100/ds4-v100-appliance.k8s.yaml
```

The template follows the existing `llm` namespace and `gpu-01` convention, uses
eight `nvidia.com/gpu` devices, mounts `/models` read-only, and runs the same
launcher contract from `/workspace/ds4`. Adjust the workspace hostPath/image
before applying it to a persistent production node.

## Probe

```bash
curl -sf http://127.0.0.1:18080/health
curl -sf http://127.0.0.1:18080/v100/status
curl -sf http://127.0.0.1:18080/metrics
```

Expected status shape:

```json
{
  "service": "ds4-v100-replay",
  "status": "ok",
  "mode": "base_one_slot",
  "readiness_level": 2,
  "mtp_enabled": false,
  "limits": {
    "slots": 1,
    "concurrent_requests": 1,
    "sequential_requests": true,
    "streaming": false,
    "external_exposure": false,
    "speculative_serving": false
  }
}
```

Expected metrics include:

```text
ds4_v100_readiness_level 2
ds4_v100_ctx_tokens 1048576
ds4_v100_mtp_enabled 0
```

With MTP verify mode enabled:

```text
DS4_V100_MTP_SERVING=verify
DS4_V100_MTP_TOP_K=5
DS4_V100_MTP_GPU=7
```

Expected status changes:

```json
{
  "mode": "mtp_verify_one_slot",
  "readiness_level": 3,
  "mtp_enabled": true,
  "limits": {
    "speculative_serving": true
  },
  "mtp": {
    "serving_mode": "verify",
    "top_k": 5,
    "gpu": 7
  }
}
```

Expected MTP metrics include:

```text
ds4_v100_mtp_enabled 1
ds4_v100_mtp_requests_total 1
ds4_v100_mtp_drafts_total 1
ds4_v100_mtp_accepted_total 1
ds4_v100_mtp_rejected_total 0
```

## Generate

```bash
curl -sf \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The next number after 15 is","tokens":2}' \
  http://127.0.0.1:18080/v100/selected-token
```

The response includes prompt token count, generated token count, selected token
bytes, stage timings, and memory/upload counters from the resident replay path.
When MTP verify mode is enabled, the response also includes an `mtp` object with
committed token, target token, draft token, top-k candidates, accept/reject
status, and draft timing.

## MTP Verify Serving Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-mtp-serving-smoke.sh \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 1 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18083 \
  --log-dir docs/sprints/drafts/SPRINT-045-MTP-SERVING
```

Expected result:

```text
first_hex=3136 mtp_accepted=1 ok
```

## Startup Throughput Benchmark

The default replay path opens the eight stage schedulers in parallel. Keep the
serial path available for before/after timing and fallback debugging:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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

The benchmark writes `serial_open.json`, `parallel_open.json`, `replay.json`,
`throughput_optimization.report`, and `throughput_optimization.json`. The report
records serial and parallel open totals, per-stage timings, speedup, decode
timing, first-token bytes, and the verdict.

## Deployment Gate

Run the production deployment smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-production-deployment-gate.sh \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 1 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18082 \
  --log-dir docs/sprints/drafts/SPRINT-045-PRODUCTION-DEPLOYMENT
```

The smoke proves:

- launcher config validation passes;
- GPU reserve checks run before upload;
- `/health`, `/v100/status`, and `/metrics` respond;
- status reports the base one-slot limits and `mtp_enabled=false`;
- the official fixture still produces first-token bytes `3136`;
- `served_requests` advances in the running service.

## Rollback

Rollback is the default mode: keep `DS4_V100_SERVE_MODE=base` and
`DS4_V100_MTP_SERVING=off`. If MTP verify mode regresses, restart the same
service with those settings. No model files or pack index need to change.

## Stop

For interactive runs, stop with `Ctrl-C`. For systemd:

```bash
sudo systemctl stop ds4-v100-appliance
```

For the Kubernetes template:

```bash
kubectl -n llm rollout restart deployment/ds4-v100-appliance
kubectl -n llm delete deployment ds4-v100-appliance
```
