# TEMP Status Report 076 - Sprint 364 Direct Input Fill

Date: 2026-05-25

## Current Focus

TP/EP compressed projection staging. Sprint 364 tested whether we can skip the
per-rank full-current staging copy before compressor/indexer dense input fill.

## What Shipped

New opt-in diagnostic gate:

- `--true-ds4-compressed-kv-direct-input-fill-gate`
- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL=1`
- `tools/ds4-v100-tp-ep-profile.py --direct-compressed-input-fill`

The gate is default-off.

## V100 Result

Shape:

```text
run mode: direct-token-major
slots: 32
context: 256K
position: 262143
decode steps: 1
```

| Variant | First token | Finite bad | Decode tok/s | Wall tok/s | Compressed-KV sum | Attn fill | Indexer fill |
|---|---:|---:|---:|---:|---:|---:|---:|
| pool+norm default | 54639 | 0 | 81.339455 | 21.006518 | 126.724613 ms | 12.587939 ms | 3.754212 ms |
| direct input fill | 54639 | 0 | 60.573442 | 19.295478 | 260.365841 ms | 84.142732 ms | 65.145245 ms |

## Decision

Reject direct compressed input fill. The peer-read path is correct, but the
remote reads are far slower than explicit staging plus local reads.

Next useful work should keep local per-rank current data and focus on reducing
local launch count or improving the compressed/indexer dense projection
kernels.

Artifacts:

```text
logs/from-cluster/sprint364-direct-compressed-input-fill-smoke/
```
