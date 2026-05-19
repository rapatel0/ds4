# DS4 V100 Base Appliance Runbook

This runbook covers the current Level 2 target: a one-slot, non-MTP DS4 V100
base appliance for operator-driven short generation on the 8x 32 GiB V100 host.
It is a correctness and usability surface, not the final throughput service.

## Scope

Supported today:

- Source-layout DSv4 Flash base model.
- 8x V100 layer-sharded resident runtime.
- One active slot.
- Sequential loopback HTTP requests.
- `/health`, `/status`, `/v100/status`.
- `POST /v100/selected-token`.
- Up to 64 generated tokens per request.

Not supported today:

- MTP forward/speculative decoding.
- Multi-slot scheduling.
- Concurrent HTTP requests.
- Streaming responses.
- OpenAI-compatible API.
- Production supervision, auth, or external network exposure.

## Build

On the V100 build pod or host:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-replay
```

For the full readiness gate:

```bash
CUDA_ARCH=sm_70 make \
  tools/ds4-v100-replay \
  tools/ds4-v100-mtp-sidecar-gate \
  tools/ds4-v100-mtp-residency-smoke
```

## Required Files

The current cluster convention is:

```text
/models/DSv4-Flash-256e-fixed.gguf
/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv
```

The base appliance only needs the source model and pack index. The MTP model is
used by the full gate to keep Level 3 readiness reporting honest.

## Start The Base Appliance

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-replay \
  --serve \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --tokens 2 \
  --host 127.0.0.1 \
  --port 18080
```

Use `--max-requests N` for bounded smoke runs. Leave it unset for an operator
session.

## Probe Health And Status

```bash
curl -sf http://127.0.0.1:18080/health
curl -sf http://127.0.0.1:18080/v100/status
```

Expected status shape:

```json
{
  "service": "ds4-v100-replay",
  "status": "ok",
  "mode": "base_one_slot",
  "readiness_level": 2,
  "mtp_enabled": false
}
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

## Smoke Test

Run the Level 2 HTTP smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-appliance-smoke.sh \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --tokens 2 \
  --requests 2 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18080 \
  --log-dir docs/sprints/drafts/SPRINT-032-APPLIANCE-LONG
```

The smoke must prove:

- `/health` returns OK.
- `/v100/status` returns `service=ds4-v100-replay` and `readiness_level=2`.
- Two sequential requests pass from one resident process.
- Each request returns the requested generated-token count.
- The official fixture still produces first-token bytes `3136`.
- Multi-token requests report nonzero continuation decode time.

## Full Gate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-032-GATE-CLUSTER-8GPU-FULL
```

For Level 2, a successful full gate should have zero failures and no
`base_appliance_usability` readiness blocker. Overall readiness may still be
`ready=false` while Level 3 reports `missing=mtp_forward`.

## Stop

If the server is running interactively, stop it with `Ctrl-C`. For bounded
smoke runs, `--max-requests` exits the server after the health/status probes and
generation requests are served.
