# DS4 V100 TP/EP NCCL + KV Matrix

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | NCCL threshold | NCCL failures |
|---|---|---|---|---|---|---|---|
| Control | 0 | 54639 | 98.076858 | 107.106917 | 1746 MiB | n/a | 0 |
| FP8 E5M2 KV | 0 | 54639 | 93.927351 | 103.344304 | 1746 MiB | n/a | 0 |
| HC-current NCCL | 14 | n/a | n/a | n/a | 1114 MiB | 1536 MiB | 5 |
| FP8 E5M2 KV + HC-current NCCL | 14 | n/a | n/a | n/a | 1114 MiB | 1536 MiB | 5 |
