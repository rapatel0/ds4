# Sprint 042 Report: Native Prompt-Token MTP Verify

## Result

`SHIP`.

Sprint 042 implemented native prompt-token MTP verify for the V100 appliance.
The MTP sidecar now drafts from the real committed target token embedding and
gpu7 post-commit target HC state. The focused and full 8-GPU gates both pass,
and readiness advances from `missing=mtp_verify` to
`missing=production_deployment`.

## Implementation Summary

- Added `ds4_v100_stage_scheduler_read_token_embedding_f32()` to expose the
  committed token embedding as F32 from BF16 `token_embd.weight` with descriptor
  and range validation.
- Added `tools/ds4-v100-mtp-forward-common.[ch]`, a tool-local resident MTP
  forward runner that accepts caller-provided `embed[4096]`, post-commit
  `prev_hc[4][4096]`, and position.
- Extended `tools/ds4-v100-mtp-verify-smoke.c` to:
  - replay the real prompt through all 8 target scheduler stages;
  - commit target token `T`;
  - read `embed(T)` and gpu7 post-commit HC;
  - run resident MTP forward from those native inputs;
  - compare MTP top-1 to target top-1 by exact token equality;
  - force a rejected draft afterward and prove target/MTP rollback parity.
- Updated `tools/ds4-v100-gate.sh` so the full gate now has an explicit
  `mtp_verify` step and no longer reports `missing=mtp_verify`.

## Focused Cluster Validation

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1500 \
  ./tools/ds4-v100-mtp-verify-smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 --ctx 1048576 \
  --report docs/sprints/drafts/SPRINT-042-MTP-VERIFY/mtp_verify.report
```

Output:

```text
mtp_verify_smoke: prompt_tokens=18 committed=926 target_top1=1 mtp_top1=1 mtp_accepted=true rejected=16 snapshot_bytes=30107648 restore_delta=0 replay_delta=0 mtp_raw_max_abs=0 PASS
```

Key report lines:

- `target_top1=1`
- `mtp_top1=1`
- `accepted=true`
- `raw_row=18`
- `n_raw=1`
- `output_weight_bytes=1059061760`
- `free_after_output_upload_bytes=17056661504`
- `snapshot_bytes=30107648`
- `restore_delta=0`
- `replay_delta=0`

Artifact:

- `docs/sprints/drafts/SPRINT-042-MTP-VERIFY/mtp_verify.report`

## Full Gate

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 \
  ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-042-GATE-CLUSTER-8GPU
```

Result:

```text
gate	mtp_verify	PASS
gate	v100_appliance_http	PASS
gate	v100_appliance_http_long	PASS
gate	readiness	NOT_READY	missing=production_deployment
gate	summary	PASS	failures=0 ready=false
```

The full gate still reports `ready=false`, but the only remaining blocker is
deployment packaging. Correctness, base HTTP loopback, MTP residency, MTP
forward, native MTP verify, rollback, replay, and timing diagnostics are all
green.

## Timing Evidence

`v100_replay_tool` remains the current timing diagnostic, not a production
throughput claim:

- prompt tokens: `18`
- generated tokens: `2`
- prompt replay: `3516.402 ms`
- continuation decode: `150.665 ms`
- generated tokens/sec: `0.544273`
- continuation tokens/sec: `6.637240`
- uploaded bytes: `156142862684`

## Remaining Blocker

The next readiness blocker is `production_deployment`: supervised service
packaging, stable configuration, health/metrics expectations, restart behavior,
operator runbook, and rollback path.
