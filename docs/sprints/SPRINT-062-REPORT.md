# Sprint 062 Report: Decode Timing And Execution-Shape Decision

## Result

`SHIP`.

Sprint 062 added an explicit decode profiling switch and captured V100 timing
evidence from the sustained serving path. The next implementation sprint should
target an opt-in stage-wavefront scheduler proof.

## Code Changes

- `tools/ds4-v100-replay.c`
  - Added `--profile-decode`.
  - Sets `DS4_V100_PROFILE_DECODE=1` only when the flag is present.
  - Reports `decode_profile` through `/v100/status`.
- `tools/ds4-v100-sustained-decode-bench.sh`
  - Added `--profile-decode`.
  - Passes the flag to the replay server.
  - Preserves averaged `stage_profile_ms` arrays in case `result.json`.
- Prior Sprint 062 checkpoint:
  - Added profile buckets from layer executor to stage scheduler to replay
    counters and JSON output.

## Validation

Local:

- `cc -fsyntax-only -I. tools/ds4-v100-replay.c`
- `bash -n tools/ds4-v100-sustained-decode-bench.sh`
- `make tools/ds4-v100-replay.o`

V100 pod:

- `CUDA_ARCH=sm_70 make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke tests/cuda_v100_selected_token_smoke tests/cuda_source_dtypes_smoke`
- `cuda_source_dtypes_smoke: ok`
- `cuda_v100_full_scheduler_smoke --slots 2: ok`
- `cuda_v100_selected_token_smoke: selected=926 expected=3136 ok`

Profiled sustained benchmark:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2400 \
  bash ./tools/ds4-v100-sustained-decode-bench.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx-tiers 1048576,262144 \
  --slot-tiers 2,4 \
  --queue-policies sequential \
  --tokens 16 \
  --requests 4 \
  --warmup-requests 1 \
  --expected-token-hex 3136 \
  --sample-ms 500 \
  --profile-decode \
  --log-dir logs/sprint062-profile
```

## Timing Evidence

| Context | Slots | Generated tok/s | Continuation tok/s | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|
| 1048576 | 2 | 3.767204 | 3.531754 | 11.000% | 20.000% |
| 1048576 | 4 | 3.732457 | 3.499179 | 11.338% | 40.000% |
| 262144 | 2 | 3.781844 | 3.545478 | 12.412% | 20.000% |
| 262144 | 4 | 3.747405 | 3.513192 | 11.182% | 40.000% |

Representative averaged `stage_profile_ms.total`:

| Context | Slots | Stage profile total sum | Stage decode sum |
|---:|---:|---:|---:|
| 1048576 | 2 | 8380.923 ms | 8397.260 ms |
| 1048576 | 4 | 16926.342 ms | 16955.221 ms |
| 262144 | 2 | 8346.975 ms | 8362.843 ms |
| 262144 | 4 | 16855.582 ms | 16885.320 ms |

The profile totals nearly equal stage decode totals, so the profiler is
capturing the material stage work. Four-slot cases do not raise aggregate tok/s;
they mainly double the serialized stage total. Context length from 256K to 1M
does not materially change this short-prompt benchmark.

## Decision

The next sprint should implement a bounded, opt-in stage-wavefront proof.

Rationale:

- The current replay path serializes stage 0 through stage 7 for each active
  microbatch.
- Stage totals dominate wall time while other GPUs are idle for most of each
  request.
- Adding slots under the current schedule increases latency but does not
  improve aggregate tok/s.
- MTP commit still needs state-safety and acceptance work, but current timing
  does not show output/MTP as the first practical throughput bottleneck.
- Low-bit kernel rewrites still matter, but profiler evidence now says the
  first larger win should come from overlapping already-correct stage work.

## Artifacts

- `logs/from-cluster/sprint062-profile/sustained_decode.tsv`
- `logs/from-cluster/sprint062-profile/sustained_decode.json`
- `logs/from-cluster/sprint062-profile/cases/*/result.json`
- `logs/from-cluster/sprint062-profile/cases/*/server_status_after.json`
