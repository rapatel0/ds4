# Sprint 373: INT8 Candidate Audit For Compressed Dense

## Overview

Build the first permanent TP/EP INT8 planning primitive for the current
compressed/indexer dense bottleneck.

Sprint 372 showed that removing diagnostic host stats materially reduces the
compressed-KV stage, but the remaining hot path is still compressed/indexer
dense projection and surrounding staging. The next format question is whether
offline INT8+scale conversion is a good path for these tensors on V100.

The contract and pack evidence matters:

- `attn_compress_{kv,gate}.weight` are BF16 source tensors in the current pack.
- `indexer.compress_{kv,gate}.weight` and `indexer.proj.weight` are BF16.
- `indexer.attn_q_b.weight` is F8 E4M3 block-128.

Therefore INT8 should not be treated as a blanket whole-model conversion. It
should be evaluated tensor-family by tensor-family.

## Scope

- Add a reusable tool that reads the TP/EP pack contract and emits an INT8
  candidate table for compressed/indexer dense tensors.
- Estimate per-TP-rank INT8+scale memory using the tc-grid layout:
  `int8 weights + fp16 per-row/per-QK scale`.
- Report the serving GEMM shape seen by the runtime:
  `M = slots`, `N = TP shard rows`, `K = input cols`.
- Classify candidates:
  - BF16 compressor/indexer tensors where INT8 can reduce memory movement.
  - F8 indexer tensors where INT8 may help compute but increases packed bytes.
  - tiny-N tensors where a standalone INT8 GEMM is unlikely to be the first
    kernel target.
- Run it against the real V100 contract and commit the artifacts.

## Out Of Scope

- Do not change runtime math yet.
- Do not convert the whole model.
- Do not vendor or wire tc-grid kernels in this sprint.
- Do not promote Sprint 372 skip-stats by default in this sprint.

## Implementation

Add:

```text
tools/ds4-v100-tp-ep-int8-candidates.c
tools/ds4-v100-tp-ep-int8-candidates
```

The tool accepts:

```text
--contract PATH
--slots N
--qk N
--out-tsv PATH
--report PATH
```

The TSV should include:

- tensor id/name
- layer
- source dtype/layout/shape
- TP rank / owning GPU
- candidate family
- `M,N,K`
- source bytes
- INT8 data bytes
- INT8 scale bytes
- INT8 total bytes
- byte delta
- kernel hint / decision note

The markdown report should include aggregate totals by family and a shape
histogram.

## Definition Of Done

- Tool builds locally.
- Tool validates argument and contract parsing errors.
- Tool runs on the V100 pod against the real TP/EP contract.
- Artifacts are copied to `logs/from-cluster/sprint373-int8-candidate-audit`.
- Docs/status are updated with the quantitative result and next decision.
- Changes are committed.

## Expected Decision

The likely next sprint is an actual INT8 workbench for the best shape family,
probably BF16 compressor matrices at:

```text
M = 32 slots
N = 128 or 64 TP shard rows
K = 4096
```

Tiny `N=8` indexer projection is probably a fusion target, not a standalone
INT8 GEMM target. F8 `indexer.attn_q_b` needs benchmarking before conversion
because INT8+scale is larger than F8 block-128 for that tensor.

## Implementation Result

Added:

```text
tools/ds4-v100-tp-ep-int8-candidates.c
tools/ds4-v100-tp-ep-int8-candidates
```

The tool reads the TP/EP pack contract and emits:

- a per-candidate TSV
- a markdown summary
- aggregate totals by family
- shape histogram by `M,N,K`
- per-GPU source versus INT8+scale byte deltas

It uses the tc-grid INT8 weight layout assumption:

```text
int8 data bytes  = shard_rows * K
scale bytes      = shard_rows * ceil(K / qk) * sizeof(fp16)
default qk       = 32
serving M        = --slots, default 32
```

## V100 Audit Result

Command:

```text
./tools/ds4-v100-tp-ep-int8-candidates \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --slots 32 \
  --qk 32 \
  --out-tsv /workspace/logs/sprint373-int8-candidate-audit/int8-candidates.tsv \
  --report /workspace/logs/sprint373-int8-candidate-audit/INT8_CANDIDATE_AUDIT.md
```

Topline:

| Rows | Source bytes | INT8+scale bytes | Delta | Source GiB | INT8 GiB |
|---:|---:|---:|---:|---:|---:|
| 1328 | 796721152 | 516112384 | -280608768 | 0.742 | 0.481 |

By family:

| Family | Rows | Source MiB | INT8 total MiB | Delta MiB | Decision |
|---|---:|---:|---:|---:|---|
| attn_compressor_bf16 | 656 | 496.000 | 263.500 | -232.500 | primary INT8 workbench target |
| indexer_compressor_bf16 | 336 | 84.000 | 44.625 | -39.375 | candidate if fused with indexer state |
| indexer_proj_tiny | 168 | 10.500 | 5.578 | -4.922 | too small for standalone GEMM; fusion target |
| indexer_q_f8 | 168 | 169.312 | 178.500 | +9.188 | compute-only candidate; not a memory win |

By shape:

| Family | DType | M | N | K | Rows | Source MiB | INT8 total MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| attn_compressor_bf16 | bf16 | 32 | 128 | 4096 | 336 | 336.000 | 178.500 |
| attn_compressor_bf16 | bf16 | 32 | 64 | 4096 | 320 | 160.000 | 85.000 |
| indexer_compressor_bf16 | bf16 | 32 | 32 | 4096 | 336 | 84.000 | 44.625 |
| indexer_proj_tiny | bf16 | 32 | 8 | 4096 | 168 | 10.500 | 5.578 |
| indexer_q_f8 | f8_e4m3_b128 | 32 | 1024 | 1024 | 168 | 169.312 | 178.500 |

Per GPU, the scoped candidate set moves from `94.977 MiB` source to
`61.525 MiB` INT8+scale, a reduction of `33.451 MiB/GPU`.

## Decision

Proceed to an INT8 workbench for the BF16 attention compressor family first:

```text
M = 32
N = 128 and 64
K = 4096
```

This is the cleanest candidate because it is a large BF16 dense path in the
current bottleneck and maps directly onto existing tc-grid V100 INT8 kernel
families. It is not a whole-model quantization decision.

Do not convert `indexer.attn_q_b.weight` to INT8 for memory. It is already F8
block-128 and INT8+scale is larger. Only benchmark it if the F8 LUT path is
demonstrably slower than an INT8 kernel enough to justify the byte increase.

## Validation

Local:

```text
make -B tools/ds4-v100-tp-ep-int8-candidates
./tools/ds4-v100-tp-ep-int8-candidates --help
malformed contract rejects with a schema error
```

V100:

```text
make -B -j80 tools/ds4-v100-tp-ep-int8-candidates
real contract audit completed
```

Artifacts:

- Cluster: `/workspace/logs/sprint373-int8-candidate-audit`
- Local: `logs/from-cluster/sprint373-int8-candidate-audit`
