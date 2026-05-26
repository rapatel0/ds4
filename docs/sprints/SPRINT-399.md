# Sprint 399: NCCL TP8 Layer Boundary Proxy

Date: 2026-05-26

## Overview

Stay on TP/EP only and continue the NCCL work in the boundary where it matches
the intended future topology: dense TP hidden-state collectives. Sprint 396
showed NCCL is much faster in an isolated collective workbench. Sprint 397
showed NCCL is the wrong backend for current compact route-indexed EP compose.
Sprint 398 showed direct remote-load fusion is the wrong shape for HC-current
staging.

This sprint extends the TP8 layer-boundary proxy with an NCCL algorithm so we
can measure the future TP hidden all-reduce boundary with resident GPU work
between collectives. This is not a PP/layer-split variant and does not touch
serving defaults.

## Constraints

- TP/EP only. No PP/layer-split work.
- No generic scheduler abstraction.
- Default behavior remains unchanged unless `--algo nccl` is requested.
- Measure `32` slot and larger microbatch shapes because single-slot decode is
  not the serving target.
- Use the V100 pod for build and benchmark.

## Implementation

Files:

- `tools/ds4-v100-tp8-layer-proxy.cu`
- `Makefile`

Planned changes:

1. Add NCCL headers, error handling, and communicator lifecycle to the TP8
   layer proxy.
2. Add `--algo nccl` to `tools/ds4-v100-tp8-layer-proxy`.
3. Implement NCCL all-reduce for the proxy's hidden-state collective.
4. Keep resident local GPU work between collectives via existing
   `--local-op-repeats`.
5. Link the proxy with `-lnccl`.

## Validation

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp8-layer-proxy
```

V100 measurements:

- `--algo doubling --tokens 32 --layers 43 --collectives-per-layer 2`
- `--algo nccl --tokens 32 --layers 43 --collectives-per-layer 2`
- same two modes with `--tokens 128`
- repeat a resident-op case with `--local-op-repeats 64`

Record:

- total boundary avg ms
- per-layer ms
- per-collective ms
- overhead-only tok/s
- effective wire GB/s
- cross-device max-abs verification

## Definition of Done

- `--algo nccl` exists and old `root`/`doubling` modes still parse.
- V100 build passes.
- NCCL proxy runs pass cross-device verification.
- Results compare NCCL against doubling at `32` and `128` tokens, including at
  least one resident-op case.
- Sprint doc, temporary status report, status, and vision are updated with the
  decision.
- Commit all kept artifacts explicitly.

## Risks

- NCCL may be faster in the proxy but still need separate serving integration
  once a true TP dense/expert boundary exists.
- The layer proxy measures hidden all-reduce, not compact EP compose. It should
  guide future TP work, not override Sprint 397's compact-compose rejection.
- Resident-op repeat count is only a proxy for real dense/expert work; use it
  to reason about boundary overhead, not as a model throughput claim.

## Outcome

Implemented `--algo nccl` in `tools/ds4-v100-tp8-layer-proxy` and linked the
target against NCCL.

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp8-layer-proxy
PASS
```

V100 matrix, `43` layers, `2` collectives/layer, `hidden=4096`, F16 payload:

| Tokens | Repeats | Algo | Avg ms | Per layer ms | Per collective ms | Eff wire GB/s | Overhead tok/s | Verify |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 32 | 0 | doubling | 29.918408 | 0.695777 | 0.347888 | 12.056 | 1069.576 | ok |
| 32 | 0 | nccl | 13.960581 | 0.324665 | 0.162332 | 22.608 | 2292.168 | ok |
| 128 | 0 | doubling | 37.313934 | 0.867766 | 0.433883 | 38.668 | 3430.354 | ok |
| 128 | 0 | nccl | 17.326618 | 0.402945 | 0.201472 | 72.864 | 7387.478 | ok |
| 32 | 64 | doubling | 28.918738 | 0.672529 | 0.336264 | 12.473 | 1106.549 | ok |
| 32 | 64 | nccl | 14.404446 | 0.334987 | 0.167494 | 21.911 | 2221.536 | ok |
| 128 | 64 | doubling | 37.140958 | 0.863743 | 0.431872 | 38.848 | 3446.330 | ok |
| 128 | 64 | nccl | 19.768570 | 0.459734 | 0.229867 | 63.863 | 6474.924 | ok |

Speedups:

- `32` tokens, no resident work: `2.14x`.
- `128` tokens, no resident work: `2.15x`.
- `32` tokens, resident work: `2.01x`.
- `128` tokens, resident work: `1.88x`.

Artifacts:

- `logs/from-cluster/sprint399-nccl-tp8-layer-proxy/`

## Decision

PROMOTE the proxy capability as the NCCL measurement path for future true TP
hidden-state collectives.

This does not change serving defaults. It does confirm that NCCL remains
materially better than the current peer-copy doubling transport when the
operation is a real TP hidden all-reduce and resident GPU work exists between
collectives. The next serving-facing step should introduce a true TP dense or
expert boundary that can use this collective shape, rather than forcing NCCL
into compact route-indexed EP compose.
