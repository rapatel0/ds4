# Sprint 238 - Layer-2 BF16 Dense Coverage Gate

Date: 2026-05-23
Status: Planned

## Overview

Sprint 237 proved packed-F8 dense compute coverage for every compatible
layer-2 F8 dense tensor group in the separate TP/EP path. The remaining layer-2
dense coverage gap is BF16 compressor/indexer math. Sprint 238 adds an
equivalent BF16 dense coverage gate: discover the layer-2 BF16 dense TP tensor
groups, load source BF16 shards from the production pack, expand BF16 inside a
CUDA kernel, and validate repeat plus bounded CPU oracle correctness.

This is still not serving and not full-layer logits equivalence. It removes the
last dense-coverage scaffold gap before composing layer-2 dataflow.

## Goals

- Keep the hard cut: no PP scheduler edits and no generic PP/TP abstraction.
- Extend the TP/EP-only full-layer smoke with `--dense-compute-all-bf16`.
- Discover layer-2 `dense_tp` rows where `source_dtype=bf16`.
- Group rows by tensor id and validate each group has eight TP shards.
- Run BF16 dense compute for every compatible layer-2 BF16 dense tensor.
- Expand BF16 values inside the GPU kernel, not in a persistent expanded
  weight arena.
- Report per-tensor:
  - tensor id;
  - rows/GPU;
  - columns;
  - BF16 bytes loaded;
  - compute ms;
  - repeat max_abs;
  - CPU oracle max_abs;
  - pass/fail.
- Preserve Sprint 237 F8 coverage mode and Sprint 235 scaffold checks.
- Run on the V100 pod at `32` slots / `256K`, MTP off.

## Non-Goals

- No PP/layer-split work.
- No changes to `ds4_v100_scheduler.*`.
- No full attention graph.
- No logits equivalence claim.
- No serving integration.
- No MTP.
- No native BF16 claim. V100 does not have native BF16 tensor cores; BF16 is
  converted inside CUDA code for this correctness gate.
- No HMMA/CUTLASS optimization yet.

## Architecture

Extend the separate TP/EP full-layer smoke:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
  --dense-compute-all-f8             # Sprint 237
  --dense-compute-all-bf16           # Sprint 238
  --dense-compute-all                # optional combined coverage alias
```

BF16 and F8 coverage should share the descriptor grouping/reporting pattern
but keep format-specific loaders and kernels simple and explicit.

## Implementation

1. Add `--dense-compute-all-bf16`.
2. Add optional `--dense-compute-all` alias for F8 plus BF16 coverage.
3. Add BF16 dense tensor discovery from the parsed contract.
4. Add BF16 shard validation:
   - source dtype is `bf16`;
   - shape is two-dimensional;
   - `shard_count=8`;
   - each rank owns `total_rows / 8` rows;
   - `bytes_estimate = rows_per_gpu * cols * 2`.
5. Add a CUDA BF16 dense kernel:

   ```text
   out[slot, local_row] =
     dot(bf16_to_f32(weight[local_row, :]), input[slot, :])
   ```

6. Add bounded CPU oracle sampling.
7. Emit one `bf16_dense_compute_tensor` line per tensor and an aggregate
   coverage summary in the final scaffold line.
8. Build and run on the V100 pod.
9. Copy evidence to
   `logs/from-cluster/sprint238-tp-bf16-dense-coverage/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | BF16 dense coverage mode |
| `docs/sprints/SPRINT-238.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint238-tp-bf16-dense-coverage/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] BF16 coverage mode is implemented only in the separate TP/EP tool.
- [ ] No PP scheduler files are modified.
- [ ] Tool discovers all layer-2 BF16 dense TP tensor groups from the real
      contract.
- [ ] Tool executes every compatible BF16 dense tensor group.
- [ ] Each executed tensor passes finite repeat checks.
- [ ] Each executed tensor passes bounded CPU oracle comparison.
- [ ] Existing full-layer scaffold, KV, EP, and F8 coverage checks still pass
      when combined coverage is enabled.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint238-tp-bf16-dense-coverage/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- BF16 compute on V100 is an explicit conversion path and will not represent
  final tensor-core throughput.
- This sprint proves BF16 dense coverage, not attention/compressor dataflow.
- CPU oracle tolerance must account for accumulation order but should remain
  tight for the deterministic small sample.

## Decision

Pending.
