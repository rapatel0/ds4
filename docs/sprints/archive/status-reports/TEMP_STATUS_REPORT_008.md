# TEMP Status Report 008

Date: 2026-05-23

## Current Topline

The appliance is not yet at the practical serving objective. The best verified
production-shaped 16-slot / 256K results are still in the `~70 tok/s`
continuation range from the per-step async pipeline era, while single-slot
filled-context synthetic decode is in the `~14-16 tok/s` continuation range.

Recent filled-context single-slot profile:

| Run | Prompt tok/s | Continuation tok/s | Key bucket |
|---|---:|---:|---|
| Sprint 191 len-1024 / ctx-262144 | `14.425868` | `14.429801` | attention `56.25%` of profiled time |

Recent topology/serving evidence:

| Direction | Best current read |
|---|---|
| Layer-split baseline | Still the only usable production path |
| Routed-only TP2 overlay | Correct but slower; do not expand |
| Single-slot attention-output HMMA | Correct but about `40%` slower |
| F8->F16 cache shortcut | Failed correctness |
| Software-pipeline stage-count variants | Tested; isolated small/neutral gains, no material served uplift |
| Full TP/EP | Not implemented yet; now the main topology candidate if we pursue TP |

## Sprint 194 Result

Added `tools/ds4-v100-tp-estimate`, which quantifies the communication envelope
for layer-split, routed-only TP2, and full TP/EP candidates.

At 16 slots / 256K / active microbatch 16:

| Topology | Total wire/token | Min transfer at 150 GB/s | Decision |
|---|---:|---:|---|
| current layer8 | `7.000 MiB` | `0.049 ms` | baseline |
| routed TP2 overlay | `21.531 MiB` | `0.151 ms` | rejected |
| full TP2/PP1 | `75.250 MiB` | `0.526 ms` | possible probe |
| full TP4/PP1 | `112.875 MiB` | `0.789 ms` | strongest TP prototype candidate |
| full TP8/PP1 | `131.688 MiB` | `0.921 ms` | later/high risk |
| TP4/PP2 hybrid | `113.875 MiB` | `0.796 ms` | memory fallback |

## Interpretation

The failed TP2 overlay does not disprove tensor parallelism. It disproves
routed-only per-layer copy-back. Full TP/EP spends more communication by design,
so it only makes sense if it also changes the compute shape: dense attention,
shared FFN, routed experts, and output ownership need to be TP/EP-native.

## Next Best Implementation

Sprint 195 should choose one of two real implementation paths:

1. Build a bounded full-layer TP4/PP1 prototype over a small layer span,
   including attention/shared/routed ownership, not just routed experts.
2. Build a monolithic routed-FFN kernel that removes the remaining global
   `mid_half` handoff between gate/up and down.

Do not continue expanding single-kernel attention toggles or the current TP2
overlay.

