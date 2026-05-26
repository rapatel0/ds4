# TEMP Status Report 399

Date: 2026-05-26

## Current Focus

TP/EP only. No PP/layer-split work.

This sprint continued NCCL work at a boundary where NCCL matches the intended
future topology: TP8 hidden-state all-reduce with resident GPU work between
collectives.

## Implemented

- Added `--algo nccl` to `tools/ds4-v100-tp8-layer-proxy`.
- Added NCCL communicator initialization/destruction to the proxy.
- Implemented NCCL F16 all-reduce for the proxy's hidden-state collective.
- Linked `tools/ds4-v100-tp8-layer-proxy` with `-lnccl`.

## Results

Build on V100: PASS.

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

NCCL speedups over doubling:

- `32` tokens, no resident work: `2.14x`.
- `128` tokens, no resident work: `2.15x`.
- `32` tokens, resident work: `2.01x`.
- `128` tokens, resident work: `1.88x`.

Artifacts:

- `logs/from-cluster/sprint399-nccl-tp8-layer-proxy/`

## Decision

Promote the proxy capability as the NCCL measurement path for true TP hidden
collectives. Do not change serving defaults yet.

This confirms NCCL is still the right transport for actual TP hidden-state
all-reduce boundaries. Sprint 397's rejection remains valid for compact
route-indexed EP compose; Sprint 399 says the next serving-facing NCCL work
needs a true TP dense/expert collective boundary to attach to.
