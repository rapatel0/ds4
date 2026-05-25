# DS4 TP/EP INT8 Candidate Audit

- Contract: `/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv`
- Slots/M estimate: `32`
- INT8 scale block: `32`

## Topline

| Rows | Source bytes | INT8+scale bytes | Delta | Source GiB | INT8 GiB |
|---:|---:|---:|---:|---:|---:|
| 1328 | 796721152 | 516112384 | -280608768 | 0.742 | 0.481 |

## By Family

| Family | Rows | Source MiB | INT8 data MiB | Scale MiB | INT8 total MiB | Delta MiB |
|---|---:|---:|---:|---:|---:|---:|
| attn_compressor_bf16 | 656 | 496.000 | 248.000 | 15.500 | 263.500 | -232.500 |
| indexer_compressor_bf16 | 336 | 84.000 | 42.000 | 2.625 | 44.625 | -39.375 |
| indexer_proj_tiny | 168 | 10.500 | 5.250 | 0.328 | 5.578 | -4.922 |
| indexer_q_f8 | 168 | 169.312 | 168.000 | 10.500 | 178.500 | 9.188 |

## By Shape

| Family | DType | M | N | K | Rows | Source MiB | INT8 total MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| attn_compressor_bf16 | bf16 | 32 | 128 | 4096 | 336 | 336.000 | 178.500 |
| indexer_compressor_bf16 | bf16 | 32 | 32 | 4096 | 336 | 84.000 | 44.625 |
| indexer_proj_tiny | bf16 | 32 | 8 | 4096 | 168 | 10.500 | 5.578 |
| indexer_q_f8 | f8_e4m3_b128 | 32 | 1024 | 1024 | 168 | 169.312 | 178.500 |
| attn_compressor_bf16 | bf16 | 32 | 64 | 4096 | 320 | 160.000 | 85.000 |

## By GPU

| GPU | Source MiB | INT8 total MiB | Delta MiB |
|---:|---:|---:|---:|
| 0 | 94.977 | 61.525 | -33.451 |
| 1 | 94.977 | 61.525 | -33.451 |
| 2 | 94.977 | 61.525 | -33.451 |
| 3 | 94.977 | 61.525 | -33.451 |
| 4 | 94.977 | 61.525 | -33.451 |
| 5 | 94.977 | 61.525 | -33.451 |
| 6 | 94.977 | 61.525 | -33.451 |
| 7 | 94.977 | 61.525 | -33.451 |
