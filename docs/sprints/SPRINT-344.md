---
sprint: 344
title: TP/EP Typed KV Stream-Sync Boundary
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 344 - TP/EP Typed KV Stream-Sync Boundary

## Goal

Measure whether broad device-wide synchronization around typed KV row work is
the main remaining typed serving overhead after Sprint 343's batched row API.

## Scope

TP/EP only. No PP/layer-split work. No MTP. Preserve typed KV semantics,
batched rows, quiet serving measurements, 32 slots, and 256K context. This
sprint narrows the typed KV row barriers from `cudaDeviceSynchronize()` to
the row-kernel stream synchronization under an opt-in gate.

## Definition of Done

- [x] Add an opt-in typed KV stream-sync boundary gate.
- [x] Apply it to typed raw-SWA, compressed-attention, ratio-4 indexer, and
      typed-history row boundaries.
- [x] Add appliance env and launcher support for the gate.
- [x] Extend the HTTP A/B harness to run a batched-row stream-sync candidate.
- [x] Build on the V100 pod.
- [x] Run the same-shape `32` request / `32` slot / `256K` / `8` token HTTP
      measurement against control and batched-row baseline.
- [x] Record the decision for the next sprint.

## Outcome

Added:

- binary flag: `--true-ds4-attention-typed-kv-stream-sync-gate`
- appliance env: `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC=1`
- HTTP harness variant: `--typed-stream-sync-variant batch-rows-stream-sync-quiet`

The gate narrows the typed KV row barriers from `cudaDeviceSynchronize()` to
`cudaStreamSynchronize(0)` at the typed raw-SWA, compressed-attention,
ratio-4 indexer, and history reload boundaries.

## Validation

Local:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-http-ab.py
git diff --check
```

Result: PASS.

V100 build:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Same-shape measurement:

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

Summary:

```text
case                                  server tok/s  decode tok/s  quiet  batch  stream
control                               309.709482    730.989696    0      0      0
typed-history                          70.408773     81.526539    0      0      0
typed-quiet                            74.617279     86.637976    1      0      0
typed-batch-rows-quiet                 79.794096     94.238623    1      1      0
typed-batch-rows-stream-sync-quiet     81.006809     95.558274    1      1      1
```

Final idle GPU recheck:

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

## Decision

If stream-sync materially improves typed throughput, continue removing
device-wide synchronization and move the row APIs toward explicit per-rank
streams/events. If it is flat, the remaining overhead is likely the typed row
kernels and history bookkeeping rather than host-side device barriers.

Stream-sync is essentially flat. It improves the batched-row quiet path from
`79.794096` to `81.006809` server tok/s (`+1.5%`) and from `94.238623` to
`95.558274` decode tok/s (`+1.4%`). The remaining gap to control is still
large.

Given observed low GPU utilization during these runs, the next sprint should
stop guessing from tok/s and collect Nsight evidence. The profile should
identify the top kernels, whether tensor-core/HMMA kernels are active in the
serving window, and whether the time is dominated by typed row pack/unpack,
peer reads, synchronization gaps, TurboMind expert kernels, or dense cuBLAS.

## Artifacts

- `logs/from-cluster/sprint344-typed-kv-stream-sync/cluster/summary.tsv`
- `logs/from-cluster/sprint344-typed-kv-stream-sync/cluster/final-gpu-recheck.csv`
