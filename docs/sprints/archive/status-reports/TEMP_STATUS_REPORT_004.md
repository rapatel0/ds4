# TEMP Status Report 004

Date: 2026-05-23

## Current Direction

Hard cut remains active: no PP/layer-split optimization work. Current work is
only the separate TP/EP path, with MTP deferred until TP/EP generated serving
is operational and benchmarked.

## Current TP/EP Topline

Best current TP/EP serving-shaped result:

```text
32 slots / 256K context
resident token-major all-layer serving loop
32 generated tokens/request
shared TP runtime
resident expert bindings
shared dense ops
EP+dense overlap
source-scheduled staged copies
skip self compose copy
multi-copy streams

wall generated tok/s:        669.222644
wall continuation tok/s:     690.469286
decode generated tok/s:      876.524260
decode continuation tok/s:   910.270244
```

This is still a tool-level serving loop, not the HTTP sustained-decode harness.

## Recent Sprints

| Sprint | Change | Result |
|---|---|---|
| 267 | Promoted shared TP runtime for token-major all-layer runs | `51.289549 -> 47.902324 ms/token` proxy |
| 268 | Advanced logical position per token step | `47.902324 -> 45.770462 ms/token` proxy |
| 269 | Ran longer continuous token-major gates | 32-step baseline: `39.290219 ms/token`, `814.452062` projected slot-step tok/s |
| 270 | Skipped same-GPU staged compose copies | 32-step topline: `37.912062 ms/token`, `844.058544` projected slot-step tok/s |
| 273 | Added serving-shaped generated/continuation metrics | Decode-only `875.486234` generated tok/s, but wall was still `10.612319` tok/s |
| 274 | Added direct resident serving loop | Wall `669.222644` generated tok/s / `690.469286` continuation tok/s |

## Bottleneck Read

The current measured decode bottleneck is still compose/all-to-all:

```text
32-step skip-self run:
  EP/overlap:  522.914003 ms
  compose:     689.877521 ms
```

Skipping self-copy helped, so copy scheduling matters. The remaining compose
cost is likely destination-side reduction, synchronization, and the staged
all-to-all boundary rather than self-copy bytes alone.

## Not Yet Done

- HTTP TP/EP serving is not shipped yet.
- Current measurements are tool-level serving loop results, not the standard sustained HTTP load-client report.
- MTP is still deferred.
- Dense low-bit fused production kernels are not the current bottleneck in this
  scaffold because dense is using the FP16 cache/cuBLAS fallback and reports
  inside the overlapped stage.

## Next Practical Step

Either:

1. Continue reducing compose/all-to-all in the scaffold, especially
   destination-side reduction/synchronization.
2. Bridge the token-major scaffold into generated/continuation serving so
   actual tok/s can be measured with the current TP/EP schedule.

The next high-value step is the HTTP sustained-decode harness bridge so this
backend can report the same operational artifacts as the frozen PP appliance.
