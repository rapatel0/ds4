# Sprint 043 Report: Production Deployment Package

## Result

`SHIP`.

Sprint 043 closed the `production_deployment` readiness blocker by adding an
operator-owned deployment package around the existing verified one-slot V100
base appliance. The served endpoint remains the base one-slot path; native MTP
is still a correctness-gated sidecar and is not exposed as speculative serving.

The full 8-GPU gate now passes with:

```text
gate	production_deployment	PASS
gate	readiness	NOT_READY	missing=throughput_optimization
gate	summary	PASS	failures=0 ready=false
```

## Implementation Summary

- Added `tools/ds4-v100-run-appliance.sh`, a launcher that loads an env file,
  validates model/MTP/pack-index paths, checks GPU visibility and free-memory
  reserve, records resolved startup config, and execs
  `tools/ds4-v100-replay --serve`.
- Added `deploy/v100/ds4-v100-appliance.env.example`.
- Added `deploy/v100/ds4-v100-appliance.service` for systemd supervision.
- Added `deploy/v100/ds4-v100-appliance.k8s.yaml` for the `llm`/`gpu-01`
  Kubernetes convention.
- Extended `tools/ds4-v100-replay --serve` with `/metrics` and richer
  `/v100/status` limits:
  - one slot;
  - one concurrent request;
  - sequential serving;
  - no streaming;
  - no external exposure;
  - no speculative serving.
- Hardened `tools/ds4-v100-appliance-smoke.sh` to assert the richer status
  contract.
- Added `tools/ds4-v100-production-deployment-gate.sh`, which starts the
  appliance through the launcher, probes `/health`, `/v100/status`,
  `/metrics`, sends a bounded generation request, and verifies first-token
  bytes `3136`.
- Wired the production deployment gate into `tools/ds4-v100-gate.sh`.
- Updated `docs/operations/DS4-V100-APPLIANCE.md` with launcher, config,
  supervision, deployment gate, rollback, and stop instructions.

## Local Validation

```bash
bash -n \
  tools/ds4-v100-run-appliance.sh \
  tools/ds4-v100-production-deployment-gate.sh \
  tools/ds4-v100-appliance-smoke.sh \
  tools/ds4-v100-gate.sh
```

```bash
./tools/ds4-v100-run-appliance.sh \
  --env deploy/v100/ds4-v100-appliance.env.example \
  --check \
  --allow-missing
```

```bash
./tools/ds4-v100-run-appliance.sh \
  --env deploy/v100/ds4-v100-appliance.env.example \
  --print-command \
  --allow-missing
```

```bash
cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. \
  -D_FILE_OFFSET_BITS=64 \
  -c -o /tmp/ds4-v100-replay.o tools/ds4-v100-replay.c
```

## Focused Cluster Deployment Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1200 \
  ./tools/ds4-v100-production-deployment-gate.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 1 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18082 \
  --log-dir docs/sprints/drafts/SPRINT-043-PRODUCTION-DEPLOYMENT
```

Output:

```text
ds4-v100-production-deployment-gate: request=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 ok
ds4-v100-production-deployment-gate: launcher=ok health=ok status=ok metrics=ok requests=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 ok
```

Key evidence:

- launcher config check passed on the real cluster;
- `/metrics` reported `ds4_v100_readiness_level 2`,
  `ds4_v100_ctx_tokens 1048576`, and `ds4_v100_mtp_enabled 0`;
- final status reported `mode=base_one_slot`, `mtp_enabled=false`,
  `slots=1`, `streaming=false`, and `served_requests=4`;
- generation returned token `926`, text hex `3136`, then EOS.

Artifact:

- `docs/sprints/drafts/SPRINT-043-PRODUCTION-DEPLOYMENT/`

## Full Gate

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 3600 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-043-GATE-CLUSTER-8GPU
```

Result:

```text
gate	mtp_verify	PASS
gate	v100_appliance_http	PASS
gate	v100_appliance_http_long	PASS
gate	production_deployment	PASS
gate	readiness	NOT_READY	missing=throughput_optimization
gate	summary	PASS	failures=0 ready=false
```

Artifact:

- `docs/sprints/drafts/SPRINT-043-GATE-CLUSTER-8GPU/`

## Timing Evidence

The current timing evidence is still diagnostic, not an optimized throughput
claim.

From the Sprint043 full-gate `v100_replay_tool` artifact:

- prompt tokens: `18`
- generated tokens: `2`
- fresh-process open/upload: `345242.115 ms`
- prompt replay: `3526.405 ms`
- continuation decode: `153.169 ms`
- prompt tokens/sec: `5.104348`
- continuation tokens/sec: `6.528739`
- generated tokens/sec including upload/open: `0.542424`
- uploaded bytes: `156142862684`

Focused deployment smoke open/upload was `301701.831 ms`; full-gate production
deployment open/upload was `289214.849 ms`.

## Remaining Blocker

The next readiness blocker is `throughput_optimization`: convert diagnostics
into a credible slot/context operating envelope and implement the first
targeted optimization, most likely parallel stage open/upload or a resident
decode benchmark harness.
