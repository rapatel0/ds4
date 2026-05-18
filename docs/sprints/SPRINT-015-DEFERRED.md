# SPRINT-015 Deferred Items

The following work is intentionally outside Sprint 015.

## Full Layer Execution

- Attention, residual, RMSNorm, HC transforms, and layer-to-layer relay remain
  deferred. Sprint 015 is bounded to FFN compute from real descriptors.

## Real Router Scheduling

- Router logits, hash/bias selection, top-k scheduling, and multi-slot expert
  grouping remain deferred. Sprint 015 may use one fixed expert to prove
  descriptor-bound real-byte compute.

## Output-Head Logits

- The BF16 output head remains covered by Sprint 012/013 bounded smokes. Sprint
  015 does not cross from GPU 0 layer-2 FFN output to GPU 7 output-head logits.

## Production Expert Kernel

- TurboMind/tc-grid style grouped expert kernels and tensor-scheduled expert
  dispatch remain deferred until descriptor-bound correctness is established.

## Serving, MTP, And Throughput

- Public serving unlock, MTP speculative decoding, multi-slot scheduling, and
  performance benchmarks remain behind full descriptor-bound layer execution.
