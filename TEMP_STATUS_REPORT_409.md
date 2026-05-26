# TEMP Status Report 409

Date: 2026-05-26

## Current Focus

TP/EP only. The active workstream is NCCL admission and throughput for the
`32` slot / `256K` serving shape.

## What Changed

Added an explicit gate to skip the unused TP-runtime compressed-state arena:

- binary flag: `--tp-runtime-skip-unused-comp-state-gate`
- launcher env: `DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE=1`
- profile flag: `--skip-tp-runtime-comp-state`

The launcher and profile harness now default the skip on. The low-level binary
still requires the explicit flag.

## Why It Matters

Before this sprint, the TP runtime allocated:

```text
comp_state_bytes_per_gpu = 1803550720
```

In the current TP runtime implementation, that pointer was only allocated,
reported, and freed. The active compressed-attention state is still in the
`RankState` mirrors, so this was pure VRAM pressure on the serving path.

## V100 Results

| Case | Result | First token | Decode tok/s | Continuation tok/s | Post-close free | NCCL reserve |
|---|---:|---:|---:|---:|---:|---|
| Direct HC-current NCCL + lazy output-head | returncode 0 | 54639 | 95.402649 | 106.596995 | 2242 MiB | pass |
| HTTP HC-current NCCL + lazy output-head | 32/32 HTTP 200 | 83480 | 113.117381 | 114.092661 | 2240 MiB | pass |
| HTTP sampled repeat | 32/32 HTTP 200 | 83480 | 114.199600 | 113.663353 | 2240 MiB | pass |

HTTP response 0 generated token sequence `[83480, 79768]`.

The sampled HTTP repeat passed the readiness checker with GPU samples,
resident KV metadata, typed KV metadata, compact MoE, checksums,
`vram_failures=0`, and `2106 MiB` minimum free VRAM.

## Memory Delta

| Checkpoint | Before | After |
|---|---:|---:|
| TP runtime comp-state | 1803550720 B/GPU | 0 B/GPU |
| `after_tp_runtime` min free | 22720 MiB | 24440 MiB |
| `after_hc_controls` min free | 1248 MiB | 2968 MiB |
| `after_lazy_output_head_close` min free | 520-522 MiB | 2240-2242 MiB |

## Decision

Promote the skip as the launcher/profile default.

HC-current NCCL is now memory-admitted at `32` slots / `256K`. The next work
should measure and optimize throughput/default selection for the NCCL boundary,
then decide whether to promote HC-current NCCL itself or replace it with a
broader TP/EP collective.

## Artifacts

- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/direct-lazy-hc-nccl-skip-comp-state/`
- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/http-lazy-hc-nccl-skip-comp-state/`
- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/http-lazy-hc-nccl-skip-comp-state-sampled/`
- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/http-readiness-sampled.json`
