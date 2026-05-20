# Sprint 064 Report: Opt-In Served Wavefront Decode

## Outcome

`SHIP` as an opt-in diagnostic path, not as a default serving optimization.

Sprint 064 wired the Sprint 063 slot-lane wavefront primitives into the
same-length non-MTP served replay path behind `--wavefront-decode`, added the
same flag to the sustained decode benchmark launcher, and validated the path on
the 8x V100 pod. Correctness passed, but throughput regressed versus a paired
serial control, so the flag should not be promoted.

## Implementation

- Added `wavefront_decode` to `ds4_v100_replay_options`.
- Added `replay_feed_token_batch_wavefront`, which advances active slots across
  stage diagonals using the slot-span scheduler APIs.
- Preserved the existing serial path as the default.
- Added `tools/ds4-v100-replay --wavefront-decode`.
- Added `tools/ds4-v100-sustained-decode-bench.sh --wavefront-decode` and
  summary reporting.

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

Evidence:

- `logs/from-cluster/sprint064-smokes/smokes.log`
- `logs/from-cluster/sprint064-wavefront-smoke/sustained_decode.tsv`
- `logs/from-cluster/sprint064-wavefront/sustained_decode.tsv`
- `logs/from-cluster/sprint064-serial-control/sustained_decode.tsv`

## Throughput

All runs used:

- Model: `/models/DSv4-Flash-256e-fixed.gguf`
- Pack index: `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`
- Prompt: `tests/test-vectors/prompts/short_reasoning_plain.txt`
- Tokens/request: `16`
- Requests/case: `4`
- Warmup requests: `1`
- Queue policy: `sequential`
- Expected token hex: `3136`

| Context | Slots | Serial generated tok/s | Wavefront generated tok/s | Delta | Serial continuation tok/s | Wavefront continuation tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 2 | `3.855080` | `3.703159` | `-3.94%` | `3.614137` | `3.471712` |
| 1,048,576 | 4 | `3.829314` | `3.687598` | `-3.70%` | `3.589982` | `3.457123` |
| 262,144 | 2 | `3.848767` | `3.687677` | `-4.18%` | `3.608219` | `3.457198` |
| 262,144 | 4 | `3.839727` | `3.694816` | `-3.77%` | `3.599744` | `3.463890` |

The wavefront path also did not improve average GPU utilization:

| Context | Slots | Serial avg GPU util | Wavefront avg GPU util |
|---:|---:|---:|---:|
| 1,048,576 | 2 | `11.093%` | `10.912%` |
| 1,048,576 | 4 | `11.860%` | `10.464%` |
| 262,144 | 2 | `10.968%` | `10.939%` |
| 262,144 | 4 | `11.907%` | `11.597%` |

## Interpretation

The served wavefront path proves that slot-addressable scheduler APIs can be
used through the HTTP batch path without corrupting state or token selection.
It does not prove useful execution overlap.

The likely reason is that the current implementation is still a single host
thread issuing synchronous stage decode and handoff calls in diagonal order.
That changes ordering, but it does not keep multiple stage devices executing
independently enough to hide stage latency. It also adds per-slot handoff and
submission overhead relative to the simpler serial batch path.

## Decision

- Keep `--wavefront-decode` as an opt-in diagnostic flag.
- Do not enable wavefront decode by default.
- Do not spend another sprint on this exact single-threaded diagonal scheduler.
- The next high-throughput sprint should either:
  - implement a real asynchronous stage pipeline with per-stage workers/streams
    and explicit in-flight slot queues, or
  - pivot to a higher-leverage path such as true MTP draft commit or
    persistent low-bit expert kernels in the hot path.
