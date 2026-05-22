# Sprint 182 - Production Pack Profile Hook

Date: 2026-05-22

## Objective

Use the restored persistent production appliance pack from Sprint 181 as the
baseline for profiling, and remove the remaining harness friction that made
profiler runs depend on one-off command edits.

## Scope

- Let sustained decode benchmarks run through an explicit replay wrapper.
- Let sustained decode benchmarks enable the replay server CUDA profiler
  window.
- Validate the normal replay binary still works through the new harness path.
- Run a synchronized 16-slot/256K production-pack profile using
  `--profile-decode`.
- If practical, run a short profiler-wrapper check with `nvprof` using the same
  sustained benchmark path.
- Record whether the current production-pack bottleneck still points at the
  larger routed-FFN / TP-EP boundary.

## Non-Goals

- No kernel rewrite in this sprint.
- No change to production defaults.
- No promotion of MTP verify or commit.

## Definition of Done

- [x] `DS4_V100_REPLAY_BIN` can override the sustained benchmark server.
- [x] `--cuda-profiler-window` can be passed through sustained decode.
- [x] Shell validation passes.
- [x] V100 production-pack profile run passes correctness.
- [x] Profile evidence is copied into `logs/from-cluster/`.
- [x] Vision status is updated.
- [x] Changes are committed.

## Outcome

Implemented the profiling harness changes against the sustained decode
benchmark:

- `tools/ds4-v100-sustained-decode-bench.sh` accepts
  `DS4_V100_REPLAY_BIN` so the server can be launched through a wrapper.
- The same harness accepts `--cuda-profiler-window` and passes it through to
  replay.
- `tools/ds4-v100-replay-nvprof-wrapper.sh` now uses
  `DS4_V100_REPLAY_UNDERLYING_BIN` for the real replay binary so
  `DS4_V100_REPLAY_BIN=./tools/ds4-v100-replay-nvprof-wrapper.sh` does not
  recurse.

Validation:

- `bash -n tools/ds4-v100-sustained-decode-bench.sh
  tools/ds4-v100-replay-nvprof-wrapper.sh` passed.
- V100 production-pack async profile passed:
  `logs/from-cluster/sprint182-production-pack-profile/profile-decode-256k-16slot/`.
- V100 production-pack synchronized profile passed:
  `logs/from-cluster/sprint182-production-pack-profile/profile-sync-256k-16slot/`.
- V100 nvprof wrapper smoke passed:
  `logs/from-cluster/sprint182-production-pack-profile/nvprof-256k-smoke/`.

## Evidence

Async per-step 16-slot / 256K profile, closest to production serving behavior:

| Context | Slots | Tokens/request | Requests | Generated tok/s | Continuation tok/s | Match | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 262144 | 16 | 8 | 16 | `20.811726` | `18.210261` | 16/16 | `27.576%` | `63%` |

The async profile run recorded:

- `avg_async_total_ms=5831.286`
- `avg_async_host_wait_ms=5809.689`
- `avg_async_handoff_sum_ms=261.636`

Synchronized 16-slot / 256K profile, diagnostic only:

| Stage | Sum across stages |
|---|---:|
| Attention | `9315.361 ms` |
| FFN | `2223.435 ms` |
| HC attention transform | `630.391 ms` |
| HC FFN transform | `787.979 ms` |
| HC final transform | `47.113 ms` |
| Total profiled stage time | `13004.282 ms` |

The synchronized profile is much slower than production serving
(`2.398485` generated tok/s and `1.199242` continuation tok/s), but it gives
useful stage visibility.

The nvprof wrapper smoke used one slot / 256K / two generated tokens through:

```text
DS4_V100_REPLAY_BIN=./tools/ds4-v100-replay-nvprof-wrapper.sh
DS4_V100_REPLAY_UNDERLYING_BIN=./tools/ds4-v100-replay
```

It passed correctness and wrote `nvprof.log`. The short smoke is not a
throughput benchmark, but it proves the wrapper and CUDA profiler window can be
used by the sustained decode harness.

The nvprof smoke's top GPU-activity buckets were:

| Bucket | Time |
|---|---:|
| CUDA HtoD memcpy | `88.365 ms` |
| F8 E4M3 B128 matmul | `51.320 ms` |
| Grouped F8 E4M3 B128 matmul | `16.716 ms` |
| TurboMind MXFP4 GEMM | `13.815 ms` |
| F32 matmul | `5.431 ms` |
| Plain RMS norm | `3.817 ms` |
| Mixed attention decode | `3.110 ms` |

## Decision

Keep the profiling harness changes. They are small, fail-closed, and remove
the ad hoc profiler-command friction.

The important technical signal changed: at the required `>=256K` context tier,
the synchronized profile shows attention/KV work dominates visible stage time.
The FFN path still matters, but it is no longer credible to treat six-route
routed-FFN wrapper tuning as the only likely lever. The async profile also
shows large host/stage wait time.

Next sprint should target a material execution-boundary change with one of
these directions:

1. Persistent attention/KV decode boundary for 256K serving.
2. Broader persistent TP/EP ownership that changes the execution shape and
   avoids per-layer full-hidden copy-back.
3. True persistent routed-FFN boundary only if it removes global-memory
   handoffs, not another wrapper-level gate/down variation.

For practical high-throughput serving, the next stage should start from
attention/KV and topology evidence, not only routed-FFN microbenchmarks.
