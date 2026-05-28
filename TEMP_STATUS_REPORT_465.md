# TEMP Status Report 465

## Current Focus

TP/EP graph-event-order correctness. No PP/layer-split work.

## What Was Tested

Shape for all A/Bs:

```text
8 requests / 8 slots / 256K context / position 262000 / 3 decode tokens
```

Control baseline:

```text
HC-current NCCL + router/FFN rank-major
first token: 52762
server decode: ~20.3-20.6 tok/s
parity: pass
```

Graph candidates:

| Candidate | First Token | Parity | Server Decode Tok/s |
|---|---:|---:|---:|
| event-ring no replay | 57097 | 0/8 | 9.328611 |
| output rank+dense wait | 42549 | 0/8 | 9.418328 |
| output full device sync | 42549 | 0/8 | 9.088940 |

## Conclusion

The graph no-replay path is not failing because the output head reads too early.
Even full `cudaDeviceSynchronize()` on every GPU before output-head gather keeps
the wrong first token (`42549` instead of `52762`). The graph-event-order path
is producing bad intermediate state earlier inside the decode step.

## Permanent Additions

- Graph-order event rings in the full-layer smoke harness.
- Diagnostic output-head graph boundary waits.
- Default-off output-sync diagnostic:
  - `--decode-cudagraph-output-sync-gate`
  - `DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC`
  - profile/A-B harness flags.

## Next

Stop running broad serving graph A/Bs until first divergence is known. Add
per-stage checksums inside the decode step and compare eager versus
graph-event-order at the same shape.
