# Sprint 073 Report: Persistent Stage Pipeline Mailboxes

## Summary

Sprint 073 shipped an opt-in `mailbox` async pipeline mode for persistent
per-stage workers. The mode is correct and slightly faster than the old
`persistent` implementation, but it remains slower than the existing `per-step`
path. Appliance `auto` therefore stays on `per-step`.

## Implementation

- Added `DS4_V100_REPLAY_ASYNC_PIPELINE_MAILBOX`.
- Added a separate mailbox runtime in `ds4_v100_replay.c` so the Sprint 066
  `persistent` runtime remains available as a control.
- Wired `--async-pipeline-mode mailbox` through:
  - `tools/ds4-v100-replay`;
  - `tools/ds4-v100-sustained-decode-bench.sh`;
  - `tools/ds4-v100-run-appliance.sh`;
  - `tools/ds4-v100-appliance-soak.sh`.
- Kept `DS4_V100_ASYNC_PIPELINE_MODE=auto` resolving to `per-step` for
  multi-slot configs.
- Updated the appliance env example to document mailbox as opt-in pending
  evidence.

## V100 Validation

Build and shell checks:

```bash
bash -n tools/ds4-v100-sustained-decode-bench.sh
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-appliance-soak.sh
CUDA_ARCH=sm_70 make tools/ds4-v100-replay \
  tests/cuda_v100_stage_wavefront_smoke \
  tests/cuda_v100_selected_token_smoke
```

Correctness:

```text
cuda_v100_stage_wavefront_smoke: token0=16 token1=926 max_abs_slot0=0 max_abs_slot1=0 ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
```

Config/CLI:

- invalid async mode exits with `rc=2` and the error text includes `mailbox`;
- launcher `--check` accepts `DS4_V100_ASYNC_PIPELINE_MODE=mailbox`;
- short sustained smoke reports `async_pipeline_mode=mailbox` and `2/2`
  token matches.

## Throughput Matrix

Fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- context: `1048576`
- slots: `2,4`
- queue policy: `sequential`
- tokens/request: `16`
- measured requests/case: `4`
- warmup requests/case: `1`
- expected first token hex: `3136`

| Mode | Slots | Generated tok/s | Continuation tok/s | Avg GPU util | Async total ms | Wait-prev sum ms | Handoff sum ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 2 | `3.862534` | `3.621125` | `12.328%` | `0.000` | `0.000` | `0.000` |
| off | 4 | `3.801132` | `3.563561` | `11.756%` | `0.000` | `0.000` | `0.000` |
| per-step | 2 | `5.562124` | `5.214491` | `14.780%` | `5583.052` | `17905.501` | `41.363` |
| per-step | 4 | `8.649395` | `8.108808` | `19.029%` | `7066.214` | `18833.140` | `245.775` |
| persistent | 2 | `5.118536` | `4.798627` | `13.761%` | `6083.273` | `20463.285` | `60.041` |
| persistent | 4 | `7.865004` | `7.373441` | `18.116%` | `7770.067` | `22034.078` | `156.900` |
| mailbox | 2 | `5.123876` | `4.803634` | `13.936%` | `6072.623` | `420.237` | `58.426` |
| mailbox | 4 | `8.053284` | `7.549953` | `18.202%` | `7599.400` | `1610.146` | `168.069` |

## Decision

At 1M/4 slots, mailbox is:

- `2.394%` faster than old persistent;
- `6.892%` slower than per-step.

This fails the Sprint 073 default-change rule. The implementation is worth
keeping as diagnostic groundwork because it isolates stage readiness better,
but the next sprint should move below pthread scheduling: CUDA event/stream
handoff, peer-copy overlap, or a kernel-side execution change.

## Artifacts

- `logs/from-cluster/sprint073-mailbox-smoke`
- `logs/from-cluster/sprint073-ab-off`
- `logs/from-cluster/sprint073-ab-per-step`
- `logs/from-cluster/sprint073-ab-persistent`
- `logs/from-cluster/sprint073-ab-mailbox`
- `logs/from-cluster/sprint073-ab-comparison`

## Validation

- local object compile for `ds4_v100_replay.o` and `tools/ds4-v100-replay.o`
- shell syntax checks for changed scripts
- V100 build for replay and CUDA smokes
- V100 wavefront and selected-token smokes
- mailbox short sustained smoke
- 1M/2 and 1M/4 off/per-step/persistent/mailbox benchmark matrix
- JSON artifact validation
- `git diff --check`
