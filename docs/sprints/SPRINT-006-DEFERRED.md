# SPRINT-006 Deferred Items

These items are intentionally outside Sprint 006. If implementation work starts
touching them, stop and re-scope before continuing.

## Decode And Prefill

- Source-layout decode and first-token generation.
- Attention math, RoPE, SwiGLU, router scoring, FFN execution, output
  projection, token sampling, or logits comparison.
- Prompt prefill and graph/state rewind behavior.

## KV And Long Context

- KV allocation and slot-management implementation.
- `attn_kv`, `indexer_kv`, SWA append, compressed KV writes, and compression
  state updates.
- F16-vs-F8 KV runtime choice.
- Long-context admission planning beyond reserve fields and report output.

## Kernel Implementation

- FP8 dense/shared-expert kernels.
- MXFP4 or FP4 routed-expert kernels.
- INT8/INT4 expert kernel selection.
- Dequant-to-FP16 HMMA tile paths.
- Stream-aware production BF16 conversion paths.
- Any low-bit kernel performance benchmark.

## Tensor-Parallel Exceptions

- Vocab-parallel output head.
- Two-way tensor-parallel routed or shared FFN variants.
- Tensor-parallel topology alternatives to the baseline layer-sharded plan.

## Output Head And MTP

- Output-head projection execution on `gpu7`.
- MTP descriptor binding, MTP state allocation, MTP scheduling, or speculative
  decoding.
- Any MTP throughput claims.

## Serving And Operations

- `ds4-server` integration.
- HTTP/gRPC serving, request batching, slot admission, or health checks.
- Deployment packaging or startup automation.
- Throughput tuning and aggregate tok/s optimization.

## Storage And Residency Changes

- Persistent dequantized FP16/F32 weight copies.
- Host-backed, managed-memory, or SSD-backed successful runtime paths.
- Pack format changes beyond descriptor/report metadata needed by the
  Sprint 006 context smoke.
