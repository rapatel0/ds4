# Sprint 067 Report: Async Pipeline Profiling And A/B Dispatch

## Outcome

`SHIP` as the preferred opt-in async path.

Sprint 067 adds async pipeline timing counters, restores the Sprint 065
per-token-step worker shape as a same-binary diagnostic mode, and benchmarks it
against the Sprint 066 persistent worker implementation. The result is clear:
per-step async is faster on the standard V100 matrix, so the bare
`--async-pipeline-decode` flag now selects `per-step`. Persistent workers remain
available through `--async-pipeline-mode persistent` for diagnostics.

## Implementation

- Added `ds4_v100_replay_async_pipeline_mode` with:
  - `off`;
  - `persistent`;
  - `per-step`.
- Added async profiling counters to `ds4_v100_replay_counters`:
  - dispatch count;
  - total dispatch wall time;
  - setup/broadcast time;
  - host wait time;
  - final completion/synchronize time;
  - per-stage wait-for-previous-slot time;
  - per-stage device synchronize time.
- Instrumented the Sprint 066 persistent pipeline.
- Restored the Sprint 065 per-step worker dispatch as an opt-in mode.
- Added replay CLI flags:
  - `--async-pipeline-decode`: preferred opt-in mode, currently `per-step`;
  - `--async-pipeline-mode off|persistent|per-step`;
  - `--async-pipeline-per-step`.
- Added `timing_ms.async_pipeline` to replay JSON.
- Added async mode metadata and averaged async timing columns to
  `tools/ds4-v100-sustained-decode-bench.sh`.

## Validation

Local:

- `cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. -D_FILE_OFFSET_BITS=64 -c -o /tmp/ds4_v100_replay.o ds4_v100_replay.c`
- `cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. -D_FILE_OFFSET_BITS=64 -c -o /tmp/ds4_v100_replay_tool.o tools/ds4-v100-replay.c`
- `bash -n tools/ds4-v100-sustained-decode-bench.sh`
- `git diff --check`
- `make ds4_v100_replay.o tools/ds4-v100-replay.o`

V100 build:

- `CUDA_ARCH=sm_70 make tools/ds4-v100-replay tests/cuda_source_dtypes_smoke tests/cuda_v100_full_scheduler_smoke tests/cuda_v100_selected_token_smoke tests/cuda_v100_stage_wavefront_smoke`

V100 smokes:

- `cuda_source_dtypes_smoke: ok`
- `cuda_v100_full_scheduler_smoke --slots 2: ok`
- `cuda_v100_selected_token_smoke: selected=926 expected=3136 ok`
- `cuda_v100_stage_wavefront_smoke: token0=16 token1=926 max_abs_slot0=0 max_abs_slot1=0 ok`
- `--async-pipeline-decode` compatibility smoke reports
  `async_pipeline_mode per-step` and token hex `3136`.

Evidence:

- `logs/from-cluster/sprint067-build/build.log`
- `logs/from-cluster/sprint067-smokes/smokes.log`
- `logs/from-cluster/sprint067-replay-json/responses.json`
- `logs/from-cluster/sprint067-default-async-smoke/sustained_decode.tsv`
- `logs/from-cluster/sprint067-persistent-smoke/sustained_decode.tsv`
- `logs/from-cluster/sprint067-serial-control/sustained_decode.tsv`
- `logs/from-cluster/sprint067-persistent/sustained_decode.tsv`
- `logs/from-cluster/sprint067-per-step/sustained_decode.tsv`

## Throughput

All matrix runs used:

- Model: `/models/DSv4-Flash-256e-fixed.gguf`
- Pack index: `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`
- Prompt: `tests/test-vectors/prompts/short_reasoning_plain.txt`
- Tokens/request: `16`
- Requests/case: `4`
- Warmup requests: `1`
- Queue policy: `sequential`
- Expected token hex: `3136`

| Context | Slots | Serial generated tok/s | Persistent async generated tok/s | Per-step async generated tok/s | Per-step vs persistent |
|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 2 | `3.853443` | `5.106227` | `5.576155` | `+9.20%` |
| 1,048,576 | 4 | `3.822580` | `7.909776` | `8.617368` | `+8.95%` |
| 262,144 | 2 | `3.856124` | `5.148728` | `5.582098` | `+8.42%` |
| 262,144 | 4 | `3.839836` | `8.034975` | `8.619294` | `+7.27%` |

Continuation throughput:

| Context | Slots | Serial continuation tok/s | Persistent async continuation tok/s | Per-step async continuation tok/s |
|---:|---:|---:|---:|---:|
| 1,048,576 | 2 | `3.612602` | `4.787088` | `5.227646` |
| 1,048,576 | 4 | `3.583669` | `7.415415` | `8.078783` |
| 262,144 | 2 | `3.615116` | `4.826933` | `5.233217` |
| 262,144 | 4 | `3.599847` | `7.532789` | `8.080588` |

Average GPU utilization:

| Context | Slots | Serial avg GPU util | Persistent async avg GPU util | Per-step async avg GPU util |
|---:|---:|---:|---:|---:|
| 1,048,576 | 2 | `11.070%` | `13.825%` | `14.655%` |
| 1,048,576 | 4 | `11.538%` | `17.847%` | `18.205%` |
| 262,144 | 2 | `12.471%` | `14.086%` | `14.727%` |
| 262,144 | 4 | `11.424%` | `19.750%` | `18.980%` |

## Timing Interpretation

The async timing counters explain the Sprint 066 regression well enough to make
a product decision.

At 1M/4 slots:

| Mode | Async total ms | Setup ms | Host wait ms | Complete ms | Wait-prev sum ms | Handoff sum ms | Device-sync sum ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistent | `7757.013` | `0.571` | `7749.626` | `6.649` | `22062.751` | `151.684` | `11.685` |
| Per-step | `7106.058` | `22.286` | `7080.320` | `3.430` | `18952.132` | `280.217` | `7.186` |

Persistent workers save thread setup time, but lose more time in stage-to-stage
waiting. That matches the code shape: one global mutex/condition variable is
used for job start, per-slot dependency publication, failure, and all-workers
completion. Every completed stage/slot broadcasts to all workers and the
dispatcher, even though only the next stage for that slot can make progress.

Per-step workers pay about `22ms` setup in the 1M/4 case, but reduce
wait-for-previous-slot accumulation by about `3.1s` and win overall. The
regression is therefore not model math; it is persistent control-plane
contention and wakeup behavior.

## Decision

- Make the bare `--async-pipeline-decode` flag select `per-step`.
- Keep `--async-pipeline-mode persistent` available for diagnostics.
- Keep async decode opt-in rather than default until the serving runbook and
  appliance launcher explicitly select it and the next sprint validates the
  operator path.
- Do not invest more in global-CV persistent workers without replacing the
  wakeup path with targeted stage-to-stage signaling or CUDA stream/event
  handoff.
