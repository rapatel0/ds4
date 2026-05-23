# Sprint 270 - TP/EP Skip Self Compose Copy

Date: 2026-05-23
Status: Complete

## Overview

Sprint 270 reduces staged compose/all-to-all work by skipping same-GPU
compose copies on the FP32 EP-return path. The previous staged path copied all
`src -> dst` shards, including `src == dst`, even though the destination GPU
can read its own local contribution buffer directly.

This is still scaffold throughput, not generated-token serving throughput.

## Implementation

`tools/ds4-v100-tp-ep-full-layer-smoke` now supports:

```text
--skip-self-compose-copy
--copy-self-compose
```

`--skip-self-compose-copy` is the default. On the FP32 return path it skips
same-GPU staged copies and points compose at the local `d_ep_contrib_all`
slice for `src == dst`. The FP16 return path is unchanged.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Logs:

- `logs/from-cluster/sprint270-skip-self-compose-copy/cluster/copy-self-16step.log`
- `logs/from-cluster/sprint270-skip-self-compose-copy/cluster/copy-self-16step-summary.log`
- `logs/from-cluster/sprint270-skip-self-compose-copy/cluster/skip-self-16step.log`
- `logs/from-cluster/sprint270-skip-self-compose-copy/cluster/skip-self-16step-summary.log`
- `logs/from-cluster/sprint270-skip-self-compose-copy/cluster/skip-self-32step.log`
- `logs/from-cluster/sprint270-skip-self-compose-copy/cluster/skip-self-32step-summary.log`

Command shape:

```text
--slots 32 --top-k 6
--warmup 0 --iters 1
--fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose
--skip-descriptor-checks --skip-predecode-probes
--shared-expert-bindings --overlap-ep-dense --source-copy-schedule
--token-major-all-layers --all-layers
```

Results:

| Metric | Copy self, 16 steps | Skip self, 16 steps | Skip self, 32 steps |
|---|---:|---:|---:|
| Layer invocations | 688 | 688 | 1376 |
| Passing invocations | 688 | 688 | 1376 |
| Sum decode ms | 644.342845 | 616.054594 | 1213.185989 |
| ms/token proxy | 40.271428 | 38.503412 | 37.912062 |
| Projected slot-step tok/s | 794.608032 | 831.095174 | 844.058544 |
| Sum EP/overlap ms | 272.572587 | 273.420413 | 522.914003 |
| Sum compose ms | 371.558564 | 342.417467 | 689.877521 |
| Wall ms | 45689.989695 | 45744.791454 | 95981.741005 |
| Checksum | 8244145680 | 8244145680 | 8297177632 |
| Result | PASS | PASS | PASS |

## Decision

Promote skip-self compose copy as the default. The 16-step A/B improves the
token-major proxy by `4.4%` and reduces compose time by `7.8%` with checksum
preserved. The new 32-step scaffold topline is `37.912062 ms/token` proxy and
`844.058544` projected slot-step tok/s.

Compose/all-to-all remains the largest measured stage, but the remaining cost
is no longer self-copy traffic. The next compose work should target
destination-side reduction/synchronization or move to generated/continuation
serving integration to measure real tok/s.
