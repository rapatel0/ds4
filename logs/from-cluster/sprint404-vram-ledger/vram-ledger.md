# DS4 V100 TP/EP VRAM Ledger

Threshold: `1536 MiB`

## Case Summary

| Case | Return | First token | Min free | Max deficit | Failing GPUs | Decode tok/s |
|---|---:|---:|---:|---:|---|---:|
| control | 0 | 54639 | 1746 | 0 | none | 98.1 |
| hc-nccl | 14 | None | 1114 | 422 | 0,1,4,5,6 | n/a |
| fp8-kv-hc-nccl | 14 | None | 1114 | 422 | 0,1,4,5,6 | n/a |

## Allocation Metadata

| Case | HC control MiB | Output weight MiB | Output logits MiB |
|---|---:|---:|---:|
| control | 317.0 | 1010.0 | 15.8 |
| hc-nccl | 317.0 | 1010.0 | 15.8 |
| fp8-kv-hc-nccl | 317.0 | 1010.0 | 15.8 |

## Checkpoint Deltas

### control

| Checkpoint | Min free | Failures | Delta used by GPU |
|---|---:|---:|---|
| `startup` | 32182 | 0 | n/a |
| `after_dense_f16_cache` | 30458 | 0 | gpu0:1724, gpu1:1724, gpu2:1724, gpu3:1724, gpu4:1724, gpu5:1724, gpu6:1724, gpu7:1724 |
| `after_rank_buffers` | 30176 | 0 | gpu0:282, gpu1:282, gpu2:282, gpu3:282, gpu4:282, gpu5:282, gpu6:282, gpu7:282 |
| `after_tp_runtime` | 23380 | 0 | gpu0:6796, gpu1:6796, gpu2:6796, gpu3:6796, gpu4:6796, gpu5:6796, gpu6:6796, gpu7:6796 |
| `after_dense_ops` | 2252 | 0 | gpu0:21128, gpu1:21128, gpu2:21128, gpu3:21128, gpu4:21128, gpu5:21128, gpu6:21128, gpu7:21128 |
| `after_hc_controls` | 1880 | 0 | gpu0:372, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |
| `after_output_head` | 1746 | 0 | gpu0:134, gpu1:130, gpu2:130, gpu3:130, gpu4:130, gpu5:130, gpu6:130, gpu7:130 |

### hc-nccl

| Checkpoint | Min free | Failures | Delta used by GPU |
|---|---:|---:|---|
| `startup` | 32182 | 0 | n/a |
| `after_dense_f16_cache` | 30458 | 0 | gpu0:1724, gpu1:1724, gpu2:1724, gpu3:1724, gpu4:1724, gpu5:1724, gpu6:1724, gpu7:1724 |
| `after_rank_buffers` | 29514 | 0 | gpu0:916, gpu1:944, gpu2:848, gpu3:852, gpu4:924, gpu5:940, gpu6:892, gpu7:868 |
| `nccl_after_rank_buffers` | 29514 | 0 | gpu0:0, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |
| `after_tp_runtime` | 22720 | 0 | gpu0:6794, gpu1:6794, gpu2:6794, gpu3:6794, gpu4:6794, gpu5:6794, gpu6:6794, gpu7:6794 |
| `after_dense_ops` | 1592 | 0 | gpu0:21128, gpu1:21128, gpu2:21128, gpu3:21128, gpu4:21128, gpu5:21128, gpu6:21128, gpu7:21128 |
| `after_hc_controls` | 1248 | 0 | gpu0:372, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |
| `after_output_head` | 1114 | 0 | gpu0:134, gpu1:130, gpu2:130, gpu3:130, gpu4:130, gpu5:130, gpu6:130, gpu7:130 |
| `nccl_after_output_head` | 1114 | 5 | gpu0:0, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |

### fp8-kv-hc-nccl

| Checkpoint | Min free | Failures | Delta used by GPU |
|---|---:|---:|---|
| `startup` | 32182 | 0 | n/a |
| `after_dense_f16_cache` | 30458 | 0 | gpu0:1724, gpu1:1724, gpu2:1724, gpu3:1724, gpu4:1724, gpu5:1724, gpu6:1724, gpu7:1724 |
| `after_rank_buffers` | 29514 | 0 | gpu0:916, gpu1:944, gpu2:848, gpu3:852, gpu4:924, gpu5:940, gpu6:892, gpu7:868 |
| `nccl_after_rank_buffers` | 29514 | 0 | gpu0:0, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |
| `after_tp_runtime` | 22720 | 0 | gpu0:6794, gpu1:6794, gpu2:6794, gpu3:6794, gpu4:6794, gpu5:6794, gpu6:6794, gpu7:6794 |
| `after_dense_ops` | 1592 | 0 | gpu0:21128, gpu1:21128, gpu2:21128, gpu3:21128, gpu4:21128, gpu5:21128, gpu6:21128, gpu7:21128 |
| `after_hc_controls` | 1248 | 0 | gpu0:372, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |
| `after_output_head` | 1114 | 0 | gpu0:134, gpu1:130, gpu2:130, gpu3:130, gpu4:130, gpu5:130, gpu6:130, gpu7:130 |
| `nccl_after_output_head` | 1114 | 5 | gpu0:0, gpu1:0, gpu2:0, gpu3:0, gpu4:0, gpu5:0, gpu6:0, gpu7:0 |
