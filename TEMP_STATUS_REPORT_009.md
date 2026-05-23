# TEMP Status Report 009

Date: 2026-05-23

## Current Topline

The appliance is still not at the practical serving objective. The usable
production path remains the layer-scheduled appliance, with best recent
16-slot/256K continuation throughput in the `~70 tok/s` band and filled-context
single-slot decode in the `~14-16 tok/s` band.

## What Has Been Tried Numerically

| Direction | Result | Decision |
|---|---:|---|
| 16-slot/256K production async era | `~70 tok/s` continuation | current usable path |
| 256-slot/16K admission | `61.065087` generated / `57.248519` continuation tok/s | safe ceiling, not enough |
| fused gate/up TurboMind | `33.430971` vs `31.312694` generated tok/s in old 8-slot/256K A/B | shipped |
| batched attention F8 HMMA | `33.697698` vs `33.380614` generated tok/s in old 8-slot/256K A/B | shipped |
| software-pipeline gate/up stage-count | isolated `0.5811 ms` vs `0.6033 ms`, served nearly flat | diagnostic only |
| fixed 768/1536-route kernels | isolated wins, served flat/slower | diagnostic only |
| routed-only TP2 overlay | `66.385703` continuation-ish band vs `70.185744` control at 16-slot/256K | rejected |
| single-slot attention scratch | len-256 continuation `15.874062` vs `12.100086` rollback | shipped |
| single-slot attention output HMMA | len-256 continuation `8.842737` vs `14.722744` control | rejected |

## Sprint 195 Result

Implemented `tools/ds4-v100-tp4-collective-smoke`, a standalone CUDA smoke for a
four-GPU hidden-state collective. It gathers one `tokens x 4096` F32 tensor from
four GPUs to a root, reduces on the root, broadcasts the result back, and
verifies all outputs.

V100 results:

| Devices | Tokens | Avg ms | Effective wire GB/s | Verify |
|---|---:|---:|---:|---|
| 0,1,2,3 | 16 | `0.114019` | `13.795` | ok |
| 4,5,6,7 | 16 | `0.109620` | `14.348` | ok |
| 0,1,2,3 | 64 | `0.269475` | `23.347` | ok |
| 0,1,2,3 | 128 | `0.502929` | `25.019` | ok |
| 0,1,2,3 | 256 | `0.965824` | `26.056` | ok |
| 0,1,2,3 | 512 | `1.856035` | `27.118` | ok |
| 4,5,6,7 | 512 | `1.855423` | `27.127` | ok |
| 0,1,2,3 | 1024 | `3.671276` | `27.419` | ok |

## Interpretation

This proves the TP4 payload can be moved and reduced correctly on the V100
NVLink islands, but the naive root collective is only a floor. It is not good
enough to justify a production full-layer TP4 path by itself.

The next TP branch should first build or adopt a better collective inside one
four-GPU NVLink island: NCCL if acceptable, otherwise repo-owned ring/tree
peer-copy all-reduce. The alternative implementation branch remains a
monolithic routed-FFN kernel that removes the global `mid_half` boundary.

## HMMA Candidate

An HMMA candidate is a path whose tensor shape, dtype conversion, and memory
layout can plausibly use Volta tensor-core HMMA instructions. On V100 that
means low-bit stored weights may remain packed in memory, but the kernel must
expand/dequantize into FP16 fragments inside the GPU and feed HMMA with FP16
inputs and FP32 accumulation. If the shape is too skinny, too fragmented, or
requires too much reshaping/global casting, it is not a good HMMA candidate even
if the math is nominally a matrix multiply.
