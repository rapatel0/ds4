# TEMP Status Report 396: NCCL TP8 Collectives

Date: 2026-05-26

## Topline

NCCL is available on the V100 pod and is materially faster than the existing
peer-copy doubling collective workbench across the DS4 hidden-state payloads.

Representative results:

```text
32-token allreduce:        13.365976 ms ->  4.513166 ms  (2.96x)
32-token rs-ag:            31.431235 ms -> 10.282541 ms  (3.06x)
128-token reduce-scatter:  29.035444 ms ->  6.076402 ms  (4.78x)
128-token allgather:       20.682822 ms ->  6.142763 ms  (3.37x)
```

All NCCL workbench runs passed exact verification:

```text
verify max_abs=0.000000000 ok
```

## What Changed

- Added `--algo nccl` to `tools/ds4-v100-tp8-collective-workbench`.
- Linked the workbench with `-lnccl`.
- Implemented NCCL paths for:
  - all-reduce
  - reduce-scatter
  - all-gather
  - reduce-scatter + all-gather
  - expert reduce as all-reduce

The existing `root` and `doubling` peer-copy paths remain available.

## Matrix

| Mode | Tokens | Doubling avg ms | NCCL avg ms | Speedup | Doubling tok/s | NCCL tok/s |
|---|---:|---:|---:|---:|---:|---:|
| allreduce | 32 | `13.365976` | `4.513166` | `2.96x` | `2394.1` | `7090.4` |
| allreduce | 128 | `17.536469` | `7.071525` | `2.48x` | `7299.1` | `18100.8` |
| reduce-scatter | 32 | `11.937009` | `4.952330` | `2.41x` | `2680.7` | `6461.6` |
| reduce-scatter | 128 | `29.035444` | `6.076402` | `4.78x` | `4408.4` | `21065.1` |
| allgather | 32 | `17.771429` | `5.466174` | `3.25x` | `1800.6` | `5854.2` |
| allgather | 128 | `20.682822` | `6.142763` | `3.37x` | `6188.7` | `20837.5` |
| rs-ag | 32 | `31.431235` | `10.282541` | `3.06x` | `1018.1` | `3112.1` |
| rs-ag | 128 | `50.516200` | `13.098625` | `3.86x` | `2533.8` | `9772.0` |
| ep-reduce | 32 | `13.823816` | `5.315978` | `2.60x` | `2314.8` | `6019.6` |
| ep-reduce | 128 | `17.574120` | `7.829183` | `2.24x` | `7283.4` | `16349.1` |

## Decision

Proceed to serving-path NCCL integration behind a gate. The isolated
measurement is strong enough that the next sprint should stop treating NCCL as
an open question and wire it into the TP/EP hidden collective/reduction
boundary for an end-to-end serving A/B.

Artifacts:

- `logs/from-cluster/sprint396-nccl-collectives/`
