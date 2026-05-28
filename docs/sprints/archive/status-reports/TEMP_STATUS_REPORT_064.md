# TEMP Status Report 064

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 352 split the compressed-KV stage that Sprint 351 showed
was the largest pre-EP owner.

## Important Correction

The direct profiler default `position=100000` does not emit compressed rows,
so it does not exercise compressed/indexer typed KV stores. Emitted-row testing
uses `position=262143`. That must be a one-token run at `ctx=262144`; a
two-token run would advance to `262144` and fails outside the configured
context.

## Emitted-Row Baseline

```text
slots: 32
ctx: 262144
position: 262143
decode steps: 1
returncode: 0
generated tok/s decode: 81.647302
sum_decode_ms: 391.929670
sum_pre_ep_compressed_kv_ms: 129.990107
compressed_kv_emitted_layers: 41
```

## Compressed-KV Breakdown

| Stage | Time |
|---|---:|
| Indexer dense | `36.615896 ms` |
| Attention dense | `24.659453 ms` |
| Attention state/emit | `24.362932 ms` |
| Attention input fill | `12.725079 ms` |
| Indexer state/emit | `9.007686 ms` |
| Indexer gather/RoPE | `5.327741 ms` |
| Indexer typed/store/score | `4.916388 ms` |
| Attention gather | `4.428356 ms` |
| Indexer input fill | `4.051283 ms` |
| Attention typed/store | `2.161061 ms` |
| Ratio shift | `1.272814 ms` |

## Store Suppression

With `--skip-compressed-store --skip-indexer-store`:

```text
returncode: 0
generated tok/s decode: 81.733945
sum_decode_ms: 391.514201
sum_pre_ep_compressed_kv_ms: 128.338783
compressed_kv_sum_ms: 127.895362
```

This is flat. Typed stores are not the main bottleneck.

## Next Move

Target fused/shared compressor-indexer input fill and compressor state/emit
work. Store suppression is not worth promoting.

