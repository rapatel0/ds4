# SPRINT-018 Deferred Items

## Full Attention Softmax And Compressed-KV Layer Output

Sprint 018 validates real descriptor-bound attention projection/control bytes,
residual add, and norm. It does not prove full attention softmax or
compressed-KV visibility as a complete layer output.

## Real-Model Selected Token

Selected-token validation remains deferred until a fuller layer-output path can
feed the output head.

## Production Arena Reuse

The smoke may use partial arenas sized from layer-state spans. Production
resident shard arena reuse remains a later scheduler/runtime task.

## Serving, MTP, And Performance

Public serving, MTP speculative decoding, multi-slot scheduling, and throughput
benchmarking remain behind full layer correctness.
