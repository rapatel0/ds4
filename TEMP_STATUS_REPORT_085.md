# TEMP Status Report 085

Date: 2026-05-25

## Current Focus

TP/EP format selection for the compressed/indexer dense bottleneck.

## Sprint 373 Result

Added a reusable contract audit tool:

```text
tools/ds4-v100-tp-ep-int8-candidates
```

It reads the TP/EP pack contract and estimates a scoped offline
INT8+fp16-scale layout for compressed/indexer dense tensors.

## Key Finding

The current hot candidate set is mostly BF16 in the pack, not FP8:

| Family | DType | Shape seen by TP rank | Source MiB | INT8+scale MiB | Decision |
|---|---|---|---:|---:|---|
| attention compressor | BF16 | `M=32, N=128/64, K=4096` | 496.000 | 263.500 | primary INT8 workbench target |
| indexer compressor | BF16 | `M=32, N=32, K=4096` | 84.000 | 44.625 | possible fused-indexer target |
| indexer projection | BF16 | `M=32, N=8, K=4096` | 10.500 | 5.578 | too tiny for standalone GEMM |
| indexer Q projection | F8 E4M3 b128 | `M=32, N=1024, K=1024` | 169.312 | 178.500 | compute-only candidate, not memory win |

Total scoped candidate set:

```text
source:      796,721,152 bytes = 0.742 GiB
INT8+scale:  516,112,384 bytes = 0.481 GiB
savings:     280,608,768 bytes aggregate
per GPU:     94.977 MiB -> 61.525 MiB
```

## Decision

Yes, offline INT8+scale is worth testing, but not as a whole-model conversion.
The next implementation target should be a V100 INT8 workbench for BF16
attention compressor matrices:

```text
M = 32
N = 128 and 64
K = 4096
```

This maps to the tc-grid INT8 V100 kernel families and directly hits the
remaining compressed-KV dense bottleneck. `indexer.attn_q_b` should stay F8
unless benchmarking proves an INT8 compute win large enough to justify the
larger packed representation.

## Artifacts

- `logs/from-cluster/sprint373-int8-candidate-audit/INT8_CANDIDATE_AUDIT.md`
- `logs/from-cluster/sprint373-int8-candidate-audit/int8-candidates.tsv`
