# TEMP_STATUS_REPORT_056

Date: 2026-05-24

## Topline

Sprint 344 tested whether narrowing typed KV barriers from device-wide sync to
stream sync would materially improve the typed serving path.

Result: it did not. Stream-sync improved the batched-row typed path only from
`79.794096` to `81.006809` server tok/s.

## Shape

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
typed candidates: skip-current-load on
```

## Results

```text
case                                  server tok/s  decode tok/s  quiet  batch  stream
control                               309.709482    730.989696    0      0      0
typed-history                          70.408773     81.526539    0      0      0
typed-quiet                            74.617279     86.637976    1      0      0
typed-batch-rows-quiet                 79.794096     94.238623    1      1      0
typed-batch-rows-stream-sync-quiet     81.006809     95.558274    1      1      1
```

## Interpretation

Stream-sync is not the main lever:

- wall throughput improved only `+1.5%` over batched-row quiet
- decode throughput improved only `+1.4%` over batched-row quiet
- control is still `3.8x` higher by wall server tok/s
- control is still `7.6x` higher by decode tok/s

Given the observed `1-3%` GPU utilization during typed-path tests, the next
sprint should collect Nsight evidence instead of making another inferred
optimization. The question to answer is which kernels and gaps dominate:

- typed F8 row pack/unpack kernels
- peer-read typed row loads
- synchronization gaps
- dense cuBLAS
- TurboMind MXFP4 expert kernels
- missing tensor-core/HMMA use in the serving window

## Final GPU State

```text
0, 0, 32495, 0
1, 0, 32495, 0
2, 0, 32495, 0
3, 0, 32495, 0
4, 0, 32495, 0
5, 0, 32495, 0
6, 0, 32495, 0
7, 0, 32495, 0
```

## Artifact

- `logs/from-cluster/sprint344-typed-kv-stream-sync/cluster/summary.tsv`
