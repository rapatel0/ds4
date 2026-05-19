# SPRINT-032 Report

## Verdict

`SHIP`

## Summary

Sprint 032 shipped the Level 2 base appliance usability gate:

- `tools/ds4-v100-replay --serve` now documents and serves `/health`,
  `/status`, and `/v100/status` alongside `POST /v100/selected-token`.
- `/v100/status` reports the base one-slot mode, readiness level, model path,
  pack-index path, context, token limits, and served request count.
- `tools/ds4-v100-appliance-smoke.sh` now probes health/status before
  generation, asserts the generated-token count, preserves first-token hex
  validation for multi-token responses, and records per-request evidence.
- `tools/ds4-v100-gate.sh` now includes a separate
  `v100_appliance_http_long` gate for two-token sequential HTTP generation and
  reports `base_appliance_usability` only when that Level 2 evidence is missing.
- `docs/operations/DS4-V100-APPLIANCE.md` documents the current operator
  runbook and explicit limits.

The overall readiness gate remains conservative: Sprint 032 removes the Level 2
base usability blocker, while readiness still reports `missing=mtp_forward`.

## Local Validation

```text
bash -n tools/ds4-v100-appliance-smoke.sh tools/ds4-v100-gate.sh
git diff --check
```

Both passed locally before cluster execution.

Help output on the V100 pod confirms the new endpoint documentation:

```text
--serve                   run a minimal HTTP endpoint
                          GET /health, GET /v100/status,
                          POST /v100/selected-token
```

The smoke help now states:

```text
The smoke also probes GET /health and GET /v100/status before generation.
```

## V100 Evidence

Focused Level 2 HTTP smoke:

```text
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

Result:

```text
request=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 continuation_ms=143.607 ok
request=2 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 continuation_ms=142.953 ok
health=ok status=ok requests=2 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 ok
```

Focused timing evidence:

```text
open_total_ms=292974.672
prompt_replay_ms=3505.265
continuation_decode_ms=143.607
output_head_ms=6.860
prompt_tokens_per_second=5.135132
continuation_tokens_per_second=6.963440
generated_tokens_per_second=0.547080
uploaded_tensors=1328
uploaded_bytes=156142862684
```

The status artifact reports:

```json
{"service":"ds4-v100-replay","status":"ok","mode":"base_one_slot","readiness_level":2,"mtp_enabled":false,"ctx_tokens":1048576,"default_tokens":2,"max_tokens":64}
```

Full V100 gate:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-032-GATE-CLUSTER-8GPU-FULL
```

Gate result:

```text
gate	build	PASS
gate	mtp_sidecar	PASS
gate	mtp_residency	PASS
gate	context_kv	PASS
gate	full_scheduler	PASS
gate	scheduler_output_head	PASS
gate	v100_replay_tool	PASS
gate	v100_appliance_http	PASS
gate	v100_appliance_http_long	PASS
gate	readiness	NOT_READY	missing=mtp_forward
gate	summary	PASS	failures=0 ready=false
```

Replay timing from the same full gate:

```text
prompt_tokens=18
generated_tokens=2
first token id=926 hex=3136 text=16 logit=35.250885
second token id=1 text=<|end of sentence|> logit=39.3052406
open_total_ms=243612.454
prompt_replay_ms=3517.546
continuation_decode_ms=143.495
output_head_ms=6.454
prompt_tokens_per_second=5.117204
continuation_tokens_per_second=6.968890
generated_tokens_per_second=0.545328
uploaded_bytes=156142862684
```

MTP sidecar residency remains intact in the full gate:

```text
mtp_runtime	gpu	7
mtp_runtime	arena_kind	device
mtp_runtime	arena_bytes	3807601408
mtp_runtime	uploaded_tensors	32
mtp_runtime	uploaded_bytes	3807600108
mtp_runtime	spot_checks	60
mtp_runtime	free_after_upload_bytes	29937369088
mtp_runtime	PASS	resident_sidecar=1
```

## Artifacts

- `docs/sprints/drafts/SPRINT-032-APPLIANCE-LONG/`
- `docs/sprints/drafts/SPRINT-032-GATE-CLUSTER-8GPU-FULL/`

## Interpretation

Level 2 is now complete with documented limits: one-slot, sequential, loopback,
non-MTP, non-streaming, and up to 64 generated tokens per request. This is a
usable base correctness appliance, not a production deployment or throughput
claim. Startup/upload remains the dominant cost at roughly 244-293 seconds per
fresh process, while resident short continuation decode is roughly 143-148 ms
for the second generated token on the official fixture.

The next critical blocker is still Level 3: K=1 MTP forward correctness from
the resident sidecar, followed by draft/verify/rollback semantics.
