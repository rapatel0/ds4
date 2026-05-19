# Sprint 049 Report: Aggregate Throughput Envelope Evidence

## Result

`SHIP`.

## Implemented

- Added [tools/ds4-v100-aggregate-throughput.sh](/Users/ravi/repos/ds4/tools/ds4-v100-aggregate-throughput.sh):
  - starts `tools/ds4-v100-replay --serve` per case;
  - drives concurrent HTTP requests (`concurrency = active_microbatch = slots`);
  - validates first-token bytes;
  - records per-case counters and latency (`avg/p50/p95/p99`) plus aggregate tok/s;
  - supports `mtp-mode off|verify|both`.

- Updated [tools/ds4-v100-gate.sh](/Users/ravi/repos/ds4/tools/ds4-v100-gate.sh):
  - new rung: `aggregate_slot_context_throughput`;
  - new readiness key: `aggregate_slot_context_throughput`.
  - readiness now reports `READY` when no keys are missing.

## Cluster Execution

Environment:

- node: `gpu-01` via pod `llamacpp-build-8gpu`
- GPUs: `8x Tesla V100-SXM2-32GB`
- model: `/models/DSv4-Flash-256e-fixed.gguf`
- pack index: `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`

Artifacts:

- base envelope run:
  - [aggregate_throughput.tsv](/Users/ravi/repos/ds4/logs/from-cluster/sprint049/aggregate_throughput.tsv)
  - [aggregate_throughput.json](/Users/ravi/repos/ds4/logs/from-cluster/sprint049/aggregate_throughput.json)
- full slot/policy run at 256K:
  - [aggregate_throughput.tsv](/Users/ravi/repos/ds4/logs/from-cluster/sprint049-full/aggregate_throughput.tsv)
  - [aggregate_throughput.json](/Users/ravi/repos/ds4/logs/from-cluster/sprint049-full/aggregate_throughput.json)
- 1M slot/policy extremes:
  - [aggregate_throughput.tsv](/Users/ravi/repos/ds4/logs/from-cluster/sprint049-1m-extremes/aggregate_throughput.tsv)
  - [aggregate_throughput.json](/Users/ravi/repos/ds4/logs/from-cluster/sprint049-1m-extremes/aggregate_throughput.json)
- focused MTP comparison:
  - [aggregate_throughput.tsv](/Users/ravi/repos/ds4/logs/from-cluster/sprint049-mtp/aggregate_throughput.tsv)
  - [aggregate_throughput.json](/Users/ravi/repos/ds4/logs/from-cluster/sprint049-mtp/aggregate_throughput.json)

## Measured Results

Base mode (`tokens=1`, policy `sequential`, `requests=8` per case):

- ctx `262144`, slots `2`: p95 `6073.592 ms`, agg tok/s `0.375205`
- ctx `262144`, slots `4`: p95 `11051.044 ms`, agg tok/s `0.375940`
- ctx `1048576`, slots `2`: p95 `5595.244 ms`, agg tok/s `0.378336`
- ctx `1048576`, slots `4`: p95 `11305.137 ms`, agg tok/s `0.372198`

Expanded slot/policy coverage (`tokens=1`, `requests=4`):

- ctx `262144`, slots `1/2/4/8`, policies `sequential` and `reject-busy`:
  all cases returned `status_200=4`, `errors=0`, `token_mismatch=0`.
- ctx `1048576`, slots `1/8`, policies `sequential` and `reject-busy`:
  all cases returned `status_200=4`, `errors=0`, `token_mismatch=0`.

Focused MTP on/off comparison (`ctx=1048576`, slots `2`, policy `sequential`, `tokens=2`, `requests=4`):

- mode `off`: p95 `6724.824 ms`, agg tok/s `0.661487`
- mode `verify`: p95 `6313.708 ms`, agg tok/s `0.687963`, MTP attempted/accepted `4/4`

## Validation

Local checks:

```bash
bash -n tools/ds4-v100-aggregate-throughput.sh
bash -n tools/ds4-v100-gate.sh
```

Cluster execution commands were run from this workspace against pod
`llamacpp-build-8gpu`, with logs copied back under `logs/from-cluster`.

## Remaining Gap

Level 6 is still partial:

1. add 128K and 512K tier load evidence with the same slot/policy matrix;
2. move from first-token-only batching to multi-token token-step batching;
3. run a complete full-gate pass with the new throughput rung and archive the
   readiness output (`ready=true` target state).
