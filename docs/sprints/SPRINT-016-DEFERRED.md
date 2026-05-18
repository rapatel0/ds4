# SPRINT-016 Deferred Items

## Full Layer Output

Attention, residual, norm, HC transforms, and layer-to-layer relay remain
deferred. Sprint 016 is scoped to router-selected FFN compute.

## Selected-Token Logits

Output-head logits and selected-token validation remain deferred until a fuller
descriptor-bound layer slice exists.

## Serving, MTP, And Performance

Public serving, MTP speculative decoding, multi-slot scheduling, and throughput
benchmarks remain behind full layer correctness.

## Production Expert Kernel

Grouped/TurboMind/tc-grid expert kernels remain deferred. Sprint 016 continues
to use correctness-first diagnostic kernels.
