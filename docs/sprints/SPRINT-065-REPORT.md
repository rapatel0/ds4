# Sprint 065 Report: Async Stage Pipeline Decode

## Outcome

`SHIP` as an opt-in throughput path.

Sprint 065 adds `--async-pipeline-decode`, a threaded per-stage decode pipeline
for same-length non-MTP batches. Unlike Sprint 064's single-threaded diagonal
ordering, this path submits work from one host thread per V100 stage so
different GPUs can overlap across active slots.

The path is correct on the short fixture and materially faster than the paired
serial control, but it remains opt-in because workers are created per
token-step batch. The next optimization should make stage workers persistent
before considering default enablement.

## Implementation

- Added `async_pipeline_decode` to `ds4_v100_replay_options`.
- Added `replay_feed_token_batch_async_pipeline` in `ds4_v100_replay.c`.
- Added per-stage worker threads using existing slot-span scheduler APIs:
  - stage 0: `decode_token_slot_span`;
  - stages 1-7: `handoff_slot_span` then `decode_hc_slot_span`.
- Preserved default serial scheduling and the Sprint 064 `--wavefront-decode`
  diagnostic path.
- Added `tools/ds4-v100-replay --async-pipeline-decode`.
- Added `tools/ds4-v100-sustained-decode-bench.sh --async-pipeline-decode` and
  TSV reporting.
- Planned Sprint 065 in `docs/sprints/SPRINT-065.md`.

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

- `logs/from-cluster/sprint065-smokes/smokes.log`
- `logs/from-cluster/sprint065-async-smoke/sustained_decode.tsv`
- `logs/from-cluster/sprint065-async/sustained_decode.tsv`
- `logs/from-cluster/sprint065-serial-control/sustained_decode.tsv`

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

| Context | Slots | Serial generated tok/s | Async generated tok/s | Delta | Serial continuation tok/s | Async continuation tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 2 | `3.852906` | `5.571149` | `+44.60%` | `3.612100` | `5.222952` |
| 1,048,576 | 4 | `3.813005` | `8.668248` | `+127.34%` | `3.574692` | `8.126482` |
| 262,144 | 2 | `3.850032` | `5.561437` | `+44.45%` | `3.609405` | `5.213847` |
| 262,144 | 4 | `3.834804` | `8.682614` | `+126.42%` | `3.595129` | `8.139951` |

Average GPU utilization also improved:

| Context | Slots | Serial avg GPU util | Async avg GPU util |
|---:|---:|---:|---:|
| 1,048,576 | 2 | `11.273%` | `14.826%` |
| 1,048,576 | 4 | `11.128%` | `19.990%` |
| 262,144 | 2 | `10.634%` | `14.822%` |
| 262,144 | 4 | `11.680%` | `18.805%` |

## Interpretation

The result confirms that the dominant issue was not just kernel math or slot
formation. The previous layer-synchronous host loop left most GPUs idle while a
single stage advanced. Per-stage host workers allow gpu0-gpu7 to overlap across
active slots and make four slots finally scale above two slots.

The implementation is still conservative:

- Handoff uses the existing blocking `cudaMemcpyPeer` path.
- Each stage synchronizes its device before publishing a slot to the next stage.
- Workers are created per token-step batch instead of being persistent.
- MTP remains disabled for the batch path.

Those constraints explain why this is still far below the aspirational
`40-200+` tok/s practical-serving range, but it is a real execution-shape
improvement and worth continuing.

## Decision

- Keep `--async-pipeline-decode` opt-in for now.
- Continue this path in the next sprint by making stage workers persistent
  across token steps and requests.
- Do not spend more time on the single-threaded `--wavefront-decode` path.
- Keep MTP commit as a separate correctness sprint unless the persistent
  pipeline work stalls.
