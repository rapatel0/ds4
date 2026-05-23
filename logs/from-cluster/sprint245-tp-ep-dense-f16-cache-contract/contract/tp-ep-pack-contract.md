# DS4 V100 TP/EP Pack Contract

Generated from `/workspace/packs/ds4-appliance-full-tm-gated-s181`.

Topology: `PP=1`, `TP=8`, `EP=8`, KV sharded, MTP off.

Config: slots `32`, ctx `262144`, KV `f8_e4m3_b128`.

## Record Counts

- source pack rows read: `1199`
- dense TP contract rows: `4096`
- F8 dense rows eligible for FP16 cache: `2920`
- BF16 dense rows eligible for FP16 shadow: `1176`
- replicated control rows: `5496`
- EP expert rows: `688`
- KV/state rows: `840`

## Memory Summary

| GPU | Dense TP | Control | EP expert | KV | Comp | Scratch | Reserve | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 1 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 2 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 3 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 4 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 5 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 6 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |
| 7 | 1.01 | 0.31 | 17.13 | 3.40 | 1.68 | 1.50 | 2.00 | 27.02 |

## Dense FP16 Runtime Cache Admission

The base plan keeps source packed dense weights resident. Sprint 244 showed a
resident FP16/cuBLAS dense path is much faster for the representative TP/EP
layer loop, so this section estimates the memory cost of materializing dense
runtime FP16 weights on each TP rank.

| GPU | F8 packed eligible | F8->FP16 cache | BF16 packed shadowable | BF16->FP16 shadow | Keep-packed total | Replace-source total | Replace headroom vs 32 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 1 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 2 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 3 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 4 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 5 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 6 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |
| 7 | 0.69 | 1.36 | 0.32 | 0.32 | 28.71 | 27.70 | 4.30 |

## Contract Rules

- Dense low-bit tensors are TP8 sharded; PP/layer ownership is rejected.
- Small F32/I32 control and router tensors are replicated across TP ranks.
- Routed experts are EP8 sharded by expert id, `32` experts per GPU.
- KV/cache records use the corrected DS4 compression schedule.
- Dense FP16 cache admission is a runtime option, not a source-format change.
  The practical serving target replaces cacheable dense source tensors in
  VRAM with the FP16 runtime cache instead of keeping both copies.
