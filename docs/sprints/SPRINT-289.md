# Sprint 289 - TP/EP Vocab-Sharded Output Head Gate

Date: 2026-05-23

## Goal

Add a TP/EP-only output-head primitive that exercises the real DS4 output-head
tensor layout across all 8 V100s.

This sprint does not touch PP/layer-split code. It also does not yet make
`/v1/completions` emit real text, because the TP/EP serving loop still needs to
carry final HC `[slots,4,4096]` into the output head.

## Context

Sprint 288 added an OpenAI-shaped diagnostic completions endpoint, but it
explicitly remained selected-token diagnostic because prompt prefill and output
head are not wired in TP/EP. The existing PP scheduler has output-head logic,
but it assumes a single output-owning stage. TP/EP instead shards
`output.weight` across vocab on all 8 GPUs.

The useful next step is to build and time the TP/EP output-head primitive
directly:

```text
synthetic HC [slots,4,4096]
  -> HC RMS norm
  -> hc_head_fn/base/scale
  -> weighted HC collapse to [slots,4096]
  -> output_norm.weight
  -> output.weight vocab shards on 8 GPUs
  -> cross-shard top-1
```

## Implementation

- Added `--output-head-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Loads real replicated control tensors:
  - `hc_head_fn`
  - `hc_head_base`
  - `hc_head_scale`
  - `output_norm.weight`
- Loads real TP-sharded `output.weight` BF16 shards:
  - vocab `129280`
  - rows per GPU `16160`
  - aggregate output weight bytes `1059061760`
- Adds GPU kernels for:
  - synthetic HC generation
  - plain RMS norm over HC rows
  - F32 `hc_head_fn` projection
  - output HC weights
  - weighted HC sum
  - output RMS norm with F32 weights
- Reuses the existing BF16 dense kernel for scalar output projection.
- Adds an opt-in BF16-to-FP16 cuBLAS projection path under
  `--dense-f16-cublas-compose`.
- Reduces local vocab-shard logits to global top-1 on host for the gate.
- Reports cold projection time, worst per-GPU projection kernel time, host
  reduction time, first token/logit, finite checks, and checksum.

## Definition of Done

- [x] The TP/EP full-layer smoke builds on the V100 pod.
- [x] `--output-head-gate` runs against the real production pack and contract.
- [x] Scalar BF16 vocab-sharded output projection passes with finite logits and
  deterministic selected-token output.
- [x] The cuBLAS diagnostic path passes and reports separate kernel timing.
- [x] Sprint status and vision are updated with the outcome.

## Validation

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Scalar BF16 projection:

```text
./tools/ds4-v100-tp-ep-full-layer-smoke \
  --output-head-gate \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --tm-index /workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv \
  --lib /workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --warmup 0 --iters 1
```

cuBLAS diagnostic projection:

```text
./tools/ds4-v100-tp-ep-full-layer-smoke \
  --output-head-gate \
  --dense-f16-cublas-compose \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --tm-index /workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv \
  --lib /workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --warmup 0 --iters 1
```

Results:

| Path | Slots | Token | Projection ms | Worst kernel ms | Host reduce ms | Result |
|---|---:|---:|---:|---:|---:|---|
| BF16 scalar | 32 | 26803 | 2192.810195 | 7.593408 | 6.070330 | PASS |
| BF16 -> FP16 cuBLAS | 32 | 26803 | 2217.599099 | 22.116352 | 5.165721 | PASS |

Evidence:

```text
logs/from-cluster/sprint289-tp-ep-output-head-gate-scalar/cluster/
logs/from-cluster/sprint289-tp-ep-output-head-gate-cublas/cluster/
```

## Decision

Promote the output-head gate as the TP/EP contract for vocab-sharded token
selection. Do not promote the current cuBLAS diagnostic as a serving path yet:
it still includes cold BF16 upload, BF16-to-FP16 expansion, handle creation, and
serial per-GPU orchestration. The scalar kernel has a better measured worst
per-GPU kernel time in this cold gate, but the real serving target should use
resident output weights and parallel per-GPU launches before deciding final
kernel selection.

## Remaining Gap

The TP/EP serving loop still produces per-rank hidden shards, not final DS4 HC.
The next sprint should carry or reconstruct final HC `[slots,4,4096]` at the
end of the 43-layer token-major loop, then call this output-head primitive from
the HTTP completion path.
