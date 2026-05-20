# Sprint 071 Report: Exact MTP Commit Serving

## Outcome

`SHIP`.

Sprint 071 adds an opt-in exact-verified one-slot MTP commit mode. Unlike
diagnostic verify mode, commit mode runs MTP inside the generation loop and
emits an accepted draft as the committed output token. It still computes the
target verifier token first, so this establishes safe state mutation rather
than a throughput win.

## Implementation

- Added narrow one-slot incremental replay hooks:
  - `ds4_v100_replay_begin_generation`;
  - `ds4_v100_replay_feed_token_at_position`;
  - `ds4_v100_replay_select_current_token`;
  - `ds4_v100_replay_finish_generation`.
- Added `--mtp-serving commit` beside `off` and `verify`.
- Kept the one-slot guard: MTP serving still rejects
  `--active-microbatch != 1`.
- Factored the MTP draft helper so verify and commit mode share the same
  resident MTP forward path.
- Added a commit-mode one-slot generation loop in `tools/ds4-v100-replay.c`.
- Added commit counters to response JSON, status JSON, and metrics:
  - `commit_mode`;
  - `commit_applied`;
  - `attempts`;
  - `accepted_count`;
  - `rejected_count`;
  - `commit_count`;
  - `ds4_v100_mtp_committed_total`.
- Updated `tools/ds4-v100-mtp-serving-smoke.sh` with `--mode verify|commit`.
- Updated `tools/ds4-v100-run-appliance.sh` and the V100 env example to accept
  `DS4_V100_MTP_SERVING=commit`.

## Validation

Local:

- `make ds4_v100_replay.o tools/ds4-v100-replay.o`
- `bash -n tools/ds4-v100-mtp-serving-smoke.sh`
- `bash -n tools/ds4-v100-run-appliance.sh`
- `git diff --check`
- JSON parse checks for copied cluster artifacts.

V100 build:

```bash
CUDA_ARCH=sm_70 make \
  tools/ds4-v100-replay \
  tools/ds4-v100-mtp-verify-smoke
```

V100 verify-mode serving smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1800 \
  ./tools/ds4-v100-mtp-serving-smoke.sh \
  --mode verify \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 2 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18087 \
  --log-dir docs/sprints/drafts/SPRINT-071-MTP-VERIFY-SERVING
```

V100 commit-mode serving smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1800 \
  ./tools/ds4-v100-mtp-serving-smoke.sh \
  --mode commit \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 2 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18088 \
  --log-dir docs/sprints/drafts/SPRINT-071-MTP-COMMIT-SERVING
```

Additional V100 checks:

- Active-microbatch guard returns `rc=2` and reports
  `--mtp-serving currently requires --active-microbatch 1`.
- `tools/ds4-v100-run-appliance.sh --check` accepts
  `DS4_V100_MTP_SERVING=commit`.
- `tools/ds4-v100-run-appliance.sh --print-command` emits
  `--mtp-serving commit`.
- Native `tools/ds4-v100-mtp-verify-smoke` still passes.

Evidence:

- `logs/from-cluster/sprint071-mtp-verify-serving/`
- `logs/from-cluster/sprint071-mtp-commit-serving/`
- `logs/from-cluster/sprint071-baseline-vs-commit/summary.json`
- `logs/from-cluster/sprint071-baseline-vs-commit/summary.txt`
- `logs/from-cluster/sprint071-mtp-verify/mtp_verify.report`

## Result

Serving smoke summary:

| Mode | Requests | Drafts | Accepted | Committed | Draft ms | Token sequence |
|---|---:|---:|---:|---:|---|---|
| `verify` | `2` | `2` | `2` | `0` | `4.314`, `4.181` | `[926, 1]` |
| `commit` | `2` | `2` | `2` | `2` | `4.347`, `4.163` | `[926, 1]` |

Baseline-vs-commit evidence:

```text
tokens_match=True
verify_tokens=[{'id': 926, 'text_hex': '3136'}, {'id': 1, 'text_hex': '3cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e'}]
commit_tokens=[{'id': 926, 'text_hex': '3136'}, {'id': 1, 'text_hex': '3cefbd9c656e64e296816f66e2968173656e74656e6365efbd9c3e'}]
commit_mode=True commit_applied=True commit_count=1 attempts=1
status_mode=mtp_commit_one_slot status_serving_mode=commit status_committed=2
```

Native MTP verify still passes:

```text
mtp_verify_smoke: prompt_tokens=18 committed=926 target_top1=1 mtp_top1=1 mtp_accepted=true rejected=16 snapshot_bytes=30107648 restore_delta=0 replay_delta=0 mtp_raw_max_abs=0 PASS
```

## Decision

- Keep commit mode opt-in and one-slot.
- Do not claim a throughput win from Sprint 071. Exact commit still computes
  the target verifier token, so the measured draft cost remains about
  `4.1-4.3 ms`.
- The next MTP sprint should measure commit-mode throughput against verify/off
  and then decide whether to implement a safe skip-verify window, recursive
  MTP, or a different throughput lever.
- Multi-slot MTP should remain blocked until MTP scratch/raw-cache ownership is
  per-slot or otherwise isolated.
