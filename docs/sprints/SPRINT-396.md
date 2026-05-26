# Sprint 396: NCCL TP8 Collective Workbench

## Overview

Add an NCCL-backed algorithm path to the existing TP8 collective workbench and
measure it on the 8x V100 pod against the current peer-copy root/doubling
collectives.

The serving path still shows low average GPU utilization at the target
`32` slot / `256K` shape. Sprint 395 cleaned up the route upload boundary but
did not change the larger diagnosis. The next requested focus is NCCL: before
touching the serving loop, we need a controlled V100 measurement of NCCL
all-reduce, reduce-scatter, and all-gather at the hidden-state payloads DS4
would use.

## Scope

- Extend `tools/ds4-v100-tp8-collective-workbench.cu` with
  `--algo nccl`.
- Link the workbench with NCCL.
- Implement NCCL paths for:
  - `allreduce`
  - `reduce-scatter`
  - `allgather`
  - `rs-ag`
  - `ep-reduce` as all-reduce-equivalent
- Keep root/doubling peer-copy algorithms intact.
- Run V100 measurements at representative token counts, including at least
  `tokens=32` and `tokens=128`.

## Out Of Scope

- No PP/layer-split work.
- No direct serving-loop integration in this sprint.
- No CUDA graph recapture work yet.
- No MTP work.

## Definition Of Done

- `--algo nccl` is implemented and default-off.
- The workbench builds on the V100 pod with NCCL linked.
- NCCL correctness passes for the target modes.
- V100 results compare NCCL against existing root/doubling algorithms.
- Sprint docs record whether NCCL is worth moving into the serving path next.

## Risks

- NCCL may have higher launch/setup overhead than peer-copy collectives for
  the small decode payloads.
- NCCL may require environment tuning on the pod to use NVLink/P2P optimally.
- Workbench wins may not transfer directly until the serving loop has a clean
  collective abstraction.

## Execution Plan

1. Add NCCL initialization and `--algo nccl` parsing to the workbench.
2. Implement NCCL collective dispatch for existing modes.
3. Build on the V100 pod.
4. Run correctness/performance measurements for peer-copy and NCCL algorithms.
5. Record a next-step decision for serving integration.

## Outcome

Complete. Added `--algo nccl` to
`tools/ds4-v100-tp8-collective-workbench` and linked the workbench with NCCL.

Implementation:

- Added NCCL headers and error handling.
- Added `Algorithm::Nccl` and `--algo nccl`.
- Initialized one NCCL communicator per visible V100 with `ncclCommInitAll`.
- Implemented NCCL dispatch for:
  - `allreduce` via `ncclAllReduce`
  - `reduce-scatter` via `ncclReduceScatter`
  - `allgather` via `ncclAllGather`
  - `rs-ag` via reduce-scatter followed by all-gather
  - `ep-reduce` via all-reduce
- Kept existing `root` and `doubling` peer-copy algorithms intact.
- Updated the Makefile target to link `-lnccl`.

The V100 pod has NCCL available:

```text
/usr/include/nccl.h
/usr/lib/x86_64-linux-gnu/libnccl.so
libnccl-dev 2.19.3-1+cuda12.2
libnccl2    2.19.3-1+cuda12.2
```

## Validation

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp8-collective-workbench
```

Build passed.

V100 measurement matrix:

```text
devices=0,1,2,3,4,5,6,7
hidden=4096
layers=43
collectives_per_layer=1
warmup=3
iters=20
tokens=32,128
modes=allreduce,reduce-scatter,allgather,rs-ag,ep-reduce
algos=doubling,nccl
```

All runs passed `verify max_abs=0.000000000 ok`.

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

NCCL is strongly positive in the isolated TP8 collective workbench. It is not
just a bandwidth improvement; it cuts per-layer collective proxy latency by
roughly `2.2x-4.8x` at the DS4 hidden payload sizes measured here.

Next sprint should add a serving-path NCCL gate for the TP/EP hidden
collective/reduction boundary rather than continuing peer-copy collective
micro-optimizations.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint396-nccl-collectives/`
- Local:
  - `logs/from-cluster/sprint396-nccl-collectives/`
