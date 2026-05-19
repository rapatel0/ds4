# Sprint 045 Report: Production MTP Speculative Serving

## Result

`SHIP`.

Sprint 045 closes the `mtp_speculative_serving` readiness blocker by exposing
the already-gated one-token MTP verify path through the resident HTTP replay
appliance. The shipped mode is conservative: base generation still produces the
returned token sequence, MTP drafts are reported and counted only after exact
top-1 comparison against the base target token, and the base one-slot path
remains the default rollback mode.

The full 8-GPU gate now passes with:

```text
gate	mtp_speculative_serving	PASS
gate	readiness	NOT_READY	missing=aggregate_slot_context_envelope
gate	summary	PASS	failures=0 ready=false
```

## Implementation Summary

- Added narrow replay accessors for the mapped base model, mapped size,
  committed-token embedding reads, and gpu7 output HC reads.
- Extended `tools/ds4-v100-replay` with `--mtp-model`,
  `--mtp-serving off|verify`, `--mtp-top-k`, `--mtp-gpu`, and
  `--mtp-reserve-mib`.
- Added a resident MTP service inside the replay tool. It opens the gpu7 MTP
  sidecar, uploads the base `output.weight`, runs one MTP draft from the latest
  committed token plus target HC, and records request/draft/accept counters.
- Kept `DS4_V100_MTP_SERVING=off` as the launcher default and added
  `DS4_V100_MTP_SERVING=verify` for explicit MTP serving.
- Added `tools/ds4-v100-mtp-serving-smoke.sh` and wired it into the full gate
  as `mtp_speculative_serving`.
- Fixed an integration lifetime bug found by the focused cluster smoke: the
  output-head binding must remain backed by an open V100 context until
  `ds4_v100_mtp_forward_open` has copied the metadata it needs.

## Local Validation

```bash
bash -n \
  tools/ds4-v100-mtp-serving-smoke.sh \
  tools/ds4-v100-run-appliance.sh \
  tools/ds4-v100-gate.sh \
  tools/ds4-v100-appliance-smoke.sh \
  tools/ds4-v100-production-deployment-gate.sh
```

```bash
git diff --check
```

```bash
make ds4_v100_replay.o tools/ds4-v100-replay.o
```

```bash
./tools/ds4-v100-run-appliance.sh --check --allow-missing
DS4_V100_MTP_SERVING=verify \
  ./tools/ds4-v100-run-appliance.sh --print-command --allow-missing
```

All local checks passed.

## Cluster Build And CLI Check

```bash
cd /workspace/ds4-sprint045
bash -n tools/ds4-v100-mtp-serving-smoke.sh \
  tools/ds4-v100-run-appliance.sh tools/ds4-v100-gate.sh
CUDA_ARCH=sm_70 make tools/ds4-v100-replay tools/ds4-v100-mtp-verify-smoke
./tools/ds4-v100-replay --help | grep -E \
  "mtp-model|mtp-serving|mtp-top-k|mtp-gpu|mtp-reserve"
```

The CUDA build passed. The generated help includes the MTP serving flags.

## Focused MTP Serving Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1200 \
  ./tools/ds4-v100-mtp-serving-smoke.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 1 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18083 \
  --log-dir docs/sprints/drafts/SPRINT-045-MTP-SERVING
```

Result:

```text
ds4-v100-mtp-serving-smoke: request=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 mtp_draft_ms=4.686 accepted=true ok
ds4-v100-mtp-serving-smoke: health=ok status=ok metrics=ok requests=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 mtp_accepted=1 ok
```

Response highlights:

```text
committed_token	926
committed_pos	18
target_token	1
draft_token	1
accepted	true
draft_ms	4.686
raw_row	18
n_raw	1
sidecar_uploaded_bytes	3807600108
output_weight_bytes	1059061760
free_after_output_upload_bytes	17056661504
```

Artifact:

- `docs/sprints/drafts/SPRINT-045-MTP-SERVING/`

## Full Gate

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 4800 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-045-GATE-CLUSTER-8GPU
```

Result:

```text
gate	mtp_verify	PASS
gate	scheduler_output_head	PASS
gate	v100_replay_tool	PASS
gate	throughput_optimization	PASS
gate	v100_appliance_http	PASS
gate	v100_appliance_http_long	PASS
gate	production_deployment	PASS
gate	mtp_speculative_serving	PASS
gate	readiness	NOT_READY	missing=aggregate_slot_context_envelope
gate	summary	PASS	failures=0 ready=false
```

Full-gate MTP serving result:

```text
ds4-v100-mtp-serving-smoke: request=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 mtp_draft_ms=4.657 accepted=true ok
ds4-v100-mtp-serving-smoke: health=ok status=ok metrics=ok requests=1 prompt_tokens=18 generated_tokens=2 first_token=926 first_hex=3136 mtp_accepted=1 ok
```

Full-gate throughput benchmark:

```text
serial_open_ms	239128.330
parallel_open_ms	52601.591
speedup	4.546028
replay_open_ms	70183.713
prompt_replay_ms	4084.614
continuation_decode_ms	143.501
continuation_tokens_per_second	6.968582
first_token_hex	3136
verdict	PASS
```

Artifact:

- `docs/sprints/drafts/SPRINT-045-GATE-CLUSTER-8GPU/`

## Current Serving Contract

Base mode remains:

```text
DS4_V100_MTP_SERVING=off
mode=base_one_slot
mtp_enabled=false
speculative_serving=false
```

MTP verify mode is explicit:

```text
DS4_V100_MTP_SERVING=verify
mode=mtp_verify_one_slot
readiness_level=3
mtp_enabled=true
speculative_serving=true
```

The served response includes an `mtp` object with draft diagnostics. Metrics now
include `ds4_v100_mtp_requests_total`, `ds4_v100_mtp_drafts_total`,
`ds4_v100_mtp_accepted_total`, `ds4_v100_mtp_rejected_total`, and
`ds4_v100_mtp_skipped_total`.

## Remaining Blocker

The next readiness blocker is `aggregate_slot_context_envelope`. Sprint 045 does
not implement multi-slot scheduling, queueing, context-tier admission, active
microbatching, or true speculative token commit without recomputing the base
target token.
