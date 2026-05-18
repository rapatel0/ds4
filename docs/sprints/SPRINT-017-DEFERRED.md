# SPRINT-017 Deferred Items

## Attention, Residual, Norm, And HC Execution

Sprint 017 owns the layer-state contract and router-selected FFN reuse. It does
not execute a full layer output.

## Real-Model Selected Token

Selected-token validation from a descriptor-bound real-model path remains
deferred until the scheduler can produce a coherent hidden state after at least
one full layer slice.

## Production Arena Reuse

The state API may report arena spans, but the smoke can still allocate a bounded
test arena. Reusing the production resident shard arena remains a later runtime
wiring task.

## Serving, MTP, And Performance

Public serving, MTP speculative decoding, multi-slot scheduling, and throughput
benchmarking remain behind full layer correctness.
