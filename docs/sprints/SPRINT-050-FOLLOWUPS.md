# Sprint 050 Follow-Ups

## P1: Optional Throughput Expansion

- Add 128K and 512K aggregate throughput matrix runs under the same harness.
- Add larger `requests` samples to reduce variance in p95/p99.

## P2: Optional Runtime Optimization

- Extend active-microbatch execution from first-token batching to multi-token
  token-step batching.
- Re-run aggregate envelope with multi-token workloads and compare.
