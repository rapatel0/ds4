# Sprint 070 Report: Persistent MTP Forward Runtime

## Outcome

`SHIP`.

Sprint 070 moves MTP forward scratch allocation from each draft call into the
opened MTP forward object. This makes the serving verifier production-shaped
for the next step: a one-slot MTP commit path that can mutate target replay
state after an accepted draft.

The timing result is important: draft latency did not materially improve
against the prior `~4.6 ms` Sprint 045 baseline. Persistent scratch is useful
state hygiene, not the next large throughput lever by itself.

## Implementation

- Added a private persistent `mtpf_scratch` bundle to
  `tools/ds4-v100-mtp-forward-common.c`.
- Allocated MTP forward tensors and the host logits buffer once during
  `ds4_v100_mtp_forward_open`.
- Reused the same scratch tensors in `ds4_v100_mtp_forward_run_host`.
- Preserved the existing raw-cache reset behavior so exact-verify semantics do
  not change.
- Extended `ds4_v100_mtp_forward_report` with:
  - `scratch_device_bytes`;
  - `scratch_host_bytes`;
  - `run_count`.
- Surfaced the new report fields in MTP serving JSON.
- Added an MTP service mutex because the forward scratch is now shared.
- Guarded MTP serving to `--active-microbatch 1` until a true commit path has
  per-slot state semantics.
- Updated `tools/ds4-v100-mtp-serving-smoke.sh` to verify scratch counters and
  sequential `forward_run_count`.

## Validation

Local:

- `make tools/ds4-v100-mtp-forward-common.o tools/ds4-v100-replay.o`
- `bash -n tools/ds4-v100-mtp-serving-smoke.sh`
- `git diff --check`

V100 build:

```bash
CUDA_ARCH=sm_70 make \
  tools/ds4-v100-replay \
  tools/ds4-v100-mtp-verify-smoke
```

V100 focused MTP serving smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1800 \
  ./tools/ds4-v100-mtp-serving-smoke.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 1048576 \
  --tokens 2 \
  --requests 3 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18086 \
  --log-dir docs/sprints/drafts/SPRINT-070-MTP-SERVING
```

V100 MTP verify smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 1500 \
  ./tools/ds4-v100-mtp-verify-smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --gpu 7 \
  --require-gpus 8 \
  --reserve-mib 4096 \
  --ctx 1048576 \
  --report docs/sprints/drafts/SPRINT-070-MTP-VERIFY/mtp_verify.report
```

Evidence:

- `logs/from-cluster/sprint070-mtp-serving/mtp_serving_response_1.json`
- `logs/from-cluster/sprint070-mtp-serving/mtp_serving_response_2.json`
- `logs/from-cluster/sprint070-mtp-serving/mtp_serving_response_3.json`
- `logs/from-cluster/sprint070-mtp-serving/mtp_serving_final_status.json`
- `logs/from-cluster/sprint070-mtp-serving/mtp_serving_final_metrics.txt`
- `logs/from-cluster/sprint070-mtp-serving/mtp_serving_server.log`
- `logs/from-cluster/sprint070-mtp-verify/mtp_verify.report`

## Result

Focused serving smoke:

| Request | Accepted | Draft ms | Scratch device bytes | Scratch host bytes | Forward run count |
|---:|---|---:|---:|---:|---:|
| 1 | `true` | `4.800` | `1848592` | `517120` | `1` |
| 2 | `true` | `4.560` | `1848592` | `517120` | `2` |
| 3 | `true` | `4.562` | `1848592` | `517120` | `3` |

Final status:

- `mode="mtp_verify_one_slot"`;
- `ctx_tokens=1048576`;
- `generation_requests=3`;
- `mtp.requests=3`;
- `mtp.drafts=3`;
- `mtp.accepted=3`;
- `mtp.rejected=0`;
- `mtp.skipped=0`.

Native MTP verify smoke still passes:

```text
mtp_verify_smoke: prompt_tokens=18 committed=926 target_top1=1 mtp_top1=1 mtp_accepted=true rejected=16 snapshot_bytes=30107648 restore_delta=0 replay_delta=0 mtp_raw_max_abs=0 PASS
```

## Decision

- Keep persistent MTP scratch and the serving JSON counters.
- Keep MTP serving one-slot until the commit path has explicit target-state
  mutation semantics.
- Do not spend another sprint on MTP allocation cleanup for throughput. The
  measured draft latency stayed near the existing `~4.6 ms` baseline.
- Sprint 071 should implement the smallest true one-slot MTP commit API:
  select a verified draft token, commit it into target replay state, and prove
  the post-commit next-token state matches clean replay.
- Lower-overhead async handoff remains a later lever. The exploratory review
  found per-step setup is small relative to total async decode time, so a
  control-only rewrite is unlikely to beat MTP commit as the next practical
  throughput step.
