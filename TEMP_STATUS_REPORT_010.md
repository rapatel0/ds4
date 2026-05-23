# TEMP Status Report 010

Date: 2026-05-23

## Current Direction

The TP4 branch now has two measured communication primitives:

- `root`: simple gather/reduce/broadcast floor;
- `doubling`: recursive pairwise all-reduce using parallel peer exchanges.

This answers the immediate tensor-parallel question more sharply: TP4
communication can be correct and fast for larger batched payloads, but it is not
yet a decode win at the current 16-slot/256K production shape.

## Sprint 196 Data

Same-tool V100 A/B on GPUs `0,1,2,3`:

| Algo | Tokens x hidden | Avg ms | Effective wire GB/s | Verify |
|---|---:|---:|---:|---|
| root | 16 x 4096 | `0.110762` | `14.200` | ok |
| doubling | 16 x 4096 | `0.133761` | `15.678` | ok |
| root | 64 x 4096 | `0.278300` | `22.607` | ok |
| doubling | 64 x 4096 | `0.184181` | `45.545` | ok |
| root | 256 x 4096 | `0.973573` | `25.849` | ok |
| doubling | 256 x 4096 | `0.496396` | `67.596` | ok |
| root | 1024 x 4096 | `3.675847` | `27.385` | ok |
| doubling | 1024 x 4096 | `1.655687` | `81.065` | ok |

Second island 16-token check:

| Devices | Algo | Avg ms | Verify |
|---|---|---:|---|
| 4,5,6,7 | root | `0.113759` | ok |
| 4,5,6,7 | doubling | `0.130128` | ok |

## Interpretation

Recursive doubling proves that the V100 NVLink island can use more aggregate
bandwidth than the naive root path. It is materially better once payloads are
large enough.

For current decode, however, the practical payload is only
`active_microbatch=16`, so the extra phase/synchronization cost dominates.
Direct TP4 all-reduce is therefore not the next production-serving lever unless
it is fused into a larger persistent layer boundary.

## Next Best Sprint

Pivot back to the monolithic routed-FFN / persistent boundary:

- remove or reduce the global `mid_half` materialization between gate/up and
  down;
- keep packed low-bit memory layout at the boundary;
- expand into FP16 fragments inside the GPU for HMMA;
- avoid global reshapes/casts and avoid returning full hidden state until the
  larger boundary is complete.

Keep `--algo doubling` as the baseline for future TP4 prefill or batched-layer
prototypes.
