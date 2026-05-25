---
sprint: 343
title: TP/EP Batched Typed KV Row API
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 343 - TP/EP Batched Typed KV Row API

## Goal

Reduce the remaining typed-KV serving overhead by replacing per-slot typed row
store/load calls with batched slot-major row APIs.

## Scope

TP/EP only. No PP/layer-split work. No MTP. Preserve the typed KV arena and
the same `32` slot / `256K` serving shape. The new batched path is opt-in so
it can be compared directly against the Sprint 342 typed-quiet baseline.

## Definition of Done

- [x] Add batched typed KV row store/load APIs to `ds4_v100_tp_runtime`.
- [x] Wire raw-SWA, compressed-attention, ratio-4 indexer, and history reloads
      to use the batched APIs under an opt-in gate.
- [x] Add appliance env and launcher support for the batched gate.
- [x] Extend the HTTP A/B harness to run a typed batched-row candidate.
- [x] Build on the V100 pod.
- [x] Run the same-shape `32` request / `32` slot / `256K` / `8` token HTTP
      measurement against control and typed-quiet baseline.
- [x] Record the decision for the next sprint.

## Outcome

Added batched slot-major typed KV APIs:

- `ds4_v100_tp_runtime_kv_rows_store_f32_device`
- `ds4_v100_tp_runtime_kv_rows_load_f32_device`

The new APIs store/load a full slot range for one layer/position/row kind in
one runtime call. The TP/EP serving path uses them under:

- binary flag: `--true-ds4-attention-typed-kv-batch-rows-gate`
- appliance env: `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS=1`

The gate applies to raw-SWA current rows, emitted compressed-attention rows,
emitted ratio-4 indexer rows, and typed-history reloads. The HTTP A/B harness
now has `--typed-batch-rows-variant batch-rows-quiet`.

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
case                    server tok/s  decode tok/s  quiet  batch_rows
control                 303.282600    735.908031    0      0
typed-history            68.450954     79.760324    0      0
typed-quiet              73.452667     86.332914    1      0
typed-batch-rows-quiet   79.984163     95.624885    1      1
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

If batched rows materially improve typed throughput, promote the batched API
shape and continue by removing the remaining broad synchronizations. If the
batched path is flat, the bottleneck is likely synchronization/order semantics
or typed-history bookkeeping rather than the host call/kernel-launch count.

Batched rows are a real improvement but not sufficient. Versus typed-quiet,
server tok/s improved from `73.452667` to `79.984163` (`+8.9%`) and decode
tok/s improved from `86.332914` to `95.624885` (`+10.8%`). The remaining gap
to control is still `3.8x` by wall tok/s and `7.7x` by decode tok/s.

Keep the batched-row gate as the better typed KV API shape. The next sprint
should remove unnecessary broad device synchronizations around the batched row
stores/history loads, or move the typed row work onto ordered per-rank streams
so CUDA stream dependencies replace device-wide barriers.

## Artifacts

- `logs/from-cluster/sprint343-typed-kv-batch-rows/cluster/summary.tsv`
- `logs/from-cluster/sprint343-typed-kv-batch-rows/cluster/final-gpu-recheck.csv`
