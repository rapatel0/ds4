# Sprint 044 Follow-Ups

## P0: MTP Speculative Serving (resolved by Sprint 045)

The full gate now passes with `missing=mtp_speculative_serving`. Next sprint
should expose the already-gated native MTP verify path through the resident
HTTP appliance without weakening the base one-slot correctness path:

- resident MTP serving/session object;
- draft from committed target token and gpu7 target HC;
- target/MTP verify transaction;
- accept/reject rollback using the existing snapshot primitive;
- `/v100/status` and `/metrics` counters for speculative serving;
- bounded loopback smoke with `mtp_enabled=true`.

## P1: Multi-Slot Aggregate Throughput

Sprint044 optimizes cold startup/upload and measures one-slot decode. It does
not implement multi-slot scheduling. Future work should add:

- admission reports for 1/2/4/8 slots at 128K, 256K, 512K, and 1M context;
- active microbatch scheduling separate from configured slots;
- queueing or explicit concurrent-request rejection semantics;
- aggregate tok/s benchmark once slot batching is implemented.

## P1: Persistent Startup Strategy

Parallel stage open/upload reduces fresh-process startup materially, but still
leaves roughly one minute of cold start in the optimized path. Longer-term
options:

- keep the process resident under supervision instead of frequent restarts;
- investigate CUDA graph or persistent staging reuse for repeated opens;
- avoid duplicate pack parsing during open;
- add per-stage upload timing regression thresholds.

## P2: Benchmark Coverage

The benchmark should eventually broaden beyond the short official fixture:

- longer prompts;
- 8/16/64 generated-token cases;
- context tiers below 1M;
- resident repeated-request latency excluding cold open;
- MTP-on versus MTP-off comparison after speculative serving ships.
