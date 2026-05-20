# Sprint 078 Report: Opt-In Event-Ordered Stage Handoff

## Outcome

`SHIP_OPT_IN_ONLY`.

Sprint 078 added CUDA event-ordered handoff for the per-step async pipeline. The
path is correct and removes the explicit per-stage device synchronization bucket,
but the paired 1M/4-slot sustained benchmark improved generated tok/s by only
`0.12%`, below the `3%` default threshold.

## Changes

- Added opaque `ds4_gpu_event` helpers and event-waited async tensor copy.
- Added scheduler event-aware HC handoff API.
- Added replay-owned reusable stage-ready events for configured slots.
- Added `--async-event-handoff` to `tools/ds4-v100-replay`.
- Added `--async-event-handoff` pass-through to
  `tools/ds4-v100-sustained-decode-bench.sh`.
- Added `DS4_V100_ASYNC_EVENT_HANDOFF=0` deployment defaults.

## Validation

Local:

```bash
make ds4_v100_scheduler.o ds4_v100_replay.o tools/ds4-v100-replay.o tests/cuda_v100_bounded_logits_smoke.o
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-sustained-decode-bench.sh
tools/ds4-v100-run-appliance.sh --check --allow-missing --env deploy/v100/ds4-v100-appliance.env.example
git diff --check
```

V100 build:

```bash
CUDA_ARCH=sm_70 make ds4_cuda.o ds4_v100_scheduler.o ds4_v100_replay.o \
  tools/ds4-v100-replay tests/cuda_v100_selected_token_smoke \
  tests/cuda_hc_relay_smoke tests/cuda_v100_full_scheduler_smoke
```

V100 smokes:

```bash
./tests/cuda_hc_relay_smoke
./tests/cuda_v100_full_scheduler_smoke --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --token 16 --position 16 --slots 4
./tests/cuda_v100_selected_token_smoke --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --expected-token-hex 3136 --top-k 1
./tools/ds4-v100-replay --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --tokens 2 --slots 4 --active-microbatch 4 \
  --async-pipeline-mode per-step --async-event-handoff \
  --expected-token-hex 3136 --json
```

The event-handoff short replay returned token sequence `[926, 1]` with first
token bytes `3136`.

## Throughput Evidence

Fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- pack index: `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`
- context: `1048576`
- slots: `4`
- tokens/request: `16`
- measured requests: `4`
- warmup requests: `1`
- async pipeline mode: `per-step`
- expected first token hex: `3136`

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Handoff sum ms | Device sync sum ms | Avg GPU util | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default per-step | `9.147418` | `8.575704` | `6994.719` | `248.432` | `6.946` | `19.958%` | `4/4` |
| Event handoff opt-in | `9.158602` | `8.586189` | `6986.078` | `193.909` | `0.000` | `19.766%` | `4/4` |

Event handoff reduced the async handoff sum by `21.95%` and removed the explicit
device-sync bucket, but end-to-end throughput improved by only `0.12%`.

## Decision

Keep event-ordered handoff opt-in behind `--async-event-handoff` and
`DS4_V100_ASYNC_EVENT_HANDOFF=1`. It is a useful primitive for later stream work,
but not enough to justify changing the appliance default.

The next throughput sprint should pivot to kernel-side work, especially routed
MXFP4 occupancy, instead of continuing small scheduling-only changes.

## Artifacts

- `logs/from-cluster/sprint078-event-default`
- `logs/from-cluster/sprint078-event-handoff`
