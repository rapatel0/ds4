# Sprint 066 Report: Persistent Async Stage Workers

## Outcome

`SHIP` as opt-in, not default.

Sprint 066 moves the `--async-pipeline-decode` path from per-token-step worker
creation to replay-owned persistent stage workers. Correctness is preserved,
and the persistent path is still faster than the serial stage-synchronous
baseline. It did not beat Sprint 065's per-step worker path, so the result is a
useful implementation step but not a default-readiness signal.

## Implementation

- Added replay-owned `replay_pipeline_runtime` state in `ds4_v100_replay.c`.
- Started one worker thread per V100 stage during `ds4_v100_replay_open` when
  `async_pipeline_decode` is enabled.
- Stopped and joined workers in `ds4_v100_replay_close` before scheduler
  teardown.
- Replaced per-dispatch `pthread_create`/`pthread_join` with a generation-based
  persistent-worker dispatch.
- Preserved existing behavior:
  - default serial decode remains unchanged;
  - one-slot batches continue through the serial path;
  - `--wavefront-decode` remains separate;
  - MTP remains outside the async batch path.

## Validation

Local:

- `cc -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99 -I. -D_FILE_OFFSET_BITS=64 -c -o /tmp/ds4_v100_replay.o ds4_v100_replay.c`
- `make ds4_v100_replay.o tools/ds4-v100-replay.o`
- `bash -n tools/ds4-v100-sustained-decode-bench.sh`
- `git diff --check`

V100 build:

- `CUDA_ARCH=sm_70 make tools/ds4-v100-replay tests/cuda_source_dtypes_smoke tests/cuda_v100_full_scheduler_smoke tests/cuda_v100_selected_token_smoke tests/cuda_v100_stage_wavefront_smoke`

V100 smokes:

- `cuda_source_dtypes_smoke: ok`
- `cuda_v100_full_scheduler_smoke --slots 2: ok`
- `cuda_v100_selected_token_smoke: selected=926 expected=3136 ok`
- `cuda_v100_stage_wavefront_smoke: token0=16 token1=926 max_abs_slot0=0 max_abs_slot1=0 ok`

Evidence:

- `logs/from-cluster/sprint066-smokes/smokes.log`
- `logs/from-cluster/sprint066-persistent-smoke/sustained_decode.tsv`
- `logs/from-cluster/sprint066-persistent/sustained_decode.tsv`
- `logs/from-cluster/sprint066-serial-control/sustained_decode.tsv`

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

| Context | Slots | Serial generated tok/s | Persistent async generated tok/s | Delta vs serial | Sprint 065 async generated tok/s | Delta vs Sprint 065 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 2 | `3.851964` | `5.132695` | `+33.25%` | `5.571149` | `-7.87%` |
| 1,048,576 | 4 | `3.788708` | `7.942345` | `+109.63%` | `8.668248` | `-8.37%` |
| 262,144 | 2 | `3.846861` | `4.729543` | `+22.95%` | `5.561437` | `-14.96%` |
| 262,144 | 4 | `3.824987` | `8.018033` | `+109.62%` | `8.682614` | `-7.65%` |

Continuation throughput:

| Context | Slots | Serial continuation tok/s | Persistent async continuation tok/s |
|---:|---:|---:|---:|
| 1,048,576 | 2 | `3.611216` | `4.811902` |
| 1,048,576 | 4 | `3.551914` | `7.445948` |
| 262,144 | 2 | `3.606432` | `4.433946` |
| 262,144 | 4 | `3.585925` | `7.516906` |

Average GPU utilization:

| Context | Slots | Serial avg GPU util | Persistent async avg GPU util |
|---:|---:|---:|---:|
| 1,048,576 | 2 | `10.134%` | `13.982%` |
| 1,048,576 | 4 | `9.858%` | `18.139%` |
| 262,144 | 2 | `12.573%` | `12.983%` |
| 262,144 | 4 | `10.878%` | `16.995%` |

## Interpretation

The result keeps the main lesson from Sprint 065: true per-stage host
concurrency is a real throughput lever. The persistent path is more than 2x
faster than serial at four slots and raises average GPU utilization from about
`10%` to `17-18%`.

The unexpected part is that persistent workers are slower than the simpler
per-step worker version. The likely causes are in the persistent dispatch
contract rather than model math:

- all stage threads rendezvous through one mutex/condition variable per token
  step;
- each stage publishes per-slot completion through the same shared condition;
- the path still performs blocking handoff plus per-device synchronization
  before making a slot visible downstream;
- one-slot batches still fall back to serial, so short smoke runs understate
  the async path.

## Decision

- Keep `--async-pipeline-decode` opt-in.
- Keep the persistent worker code because it is correct and still improves over
  serial, but do not make it default-ready.
- Next sprint should profile and reduce persistent dispatch/handoff overhead:
  per-stage queues, fewer global broadcasts, stream/event handoff, or reverting
  to the faster Sprint 065 dispatch shape if profiling confirms the persistent
  rendezvous is the regression.
- MTP commit remains separate; this sprint did not change MTP serving.
