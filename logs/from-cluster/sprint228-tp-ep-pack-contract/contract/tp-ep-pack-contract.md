# DS4 V100 TP/EP Pack Contract

Generated from `/workspace/packs/ds4-appliance-full-tm-gated-s181`.

Topology: `PP=1`, `TP=8`, `EP=8`, KV sharded, MTP off.

Config: slots `32`, ctx `262144`, KV `f8_e4m3_b128`.

## Record Counts

- source pack rows read: `1199`
- dense TP contract rows: `4096`
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

## Contract Rules

- Dense low-bit tensors are TP8 sharded; PP/layer ownership is rejected.
- Small F32/I32 control and router tensors are replicated across TP ranks.
- Routed experts are EP8 sharded by expert id, `32` experts per GPU.
- KV/cache records use the corrected DS4 compression schedule.
