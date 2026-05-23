# Sprint 236 - Descriptor Backed TP Dense Compute Gate

Date: 2026-05-23
Status: Planned

## Overview

Sprint 235 proved that all layer-2 descriptor families can be parsed, loaded,
and touched in the separate TP/EP runtime path. Dense and control rows were
still checksum scaffolds. Sprint 236 replaces one representative dense checksum
stage with actual descriptor-backed low-bit dense computation on all eight
V100s.

The selected gate is `blk.2.attn_q_a.weight` because it is a real layer-2 F8
E4M3 block-128 dense tensor with manageable dimensions:

```text
source dtype:   f8_e4m3_b128
source shape:   [4096x1024]
TP split:       8-way tensor_dim
per GPU rows:   128
input width:    4096
target slots:   32
```

This sprint is still not a full logits-equivalent DS4 layer and not serving.
It is the first real dense-math gate inside the new TP/EP codepath.

## Goals

- Keep the hard cut: no PP scheduler edits and no generic PP/TP abstraction.
- Extend the TP/EP-only full-layer smoke with an opt-in real dense compute
  gate.
- Resolve `blk.2.attn_q_a.weight` from the real TP/EP contract.
- Load each TP rank's physical F8 shard from the production pack.
- Generate deterministic multi-slot input activations in device memory.
- Run a V100 CUDA F8 E4M3 block-128 dequant-dot kernel for:
  - `32` slots;
  - `128` rows per GPU;
  - `4096` input columns.
- Verify output is finite and deterministic across repeat launches.
- Compare a bounded CPU oracle sample against GPU output with a declared
  tolerance.
- Preserve the Sprint 235 full-layer scaffold:
  - dense/control descriptor device checks;
  - sharded KV gate;
  - descriptor-backed EP expert execution.
- Run on the V100 pod at `32` slots / `256K`, MTP off.

## Non-Goals

- No PP/layer-split work.
- No changes to `ds4_v100_scheduler.*`.
- No full attention computation yet.
- No all-dense-tensor implementation yet.
- No logits equivalence claim.
- No serving integration.
- No MTP.
- No HMMA/CUTLASS optimization yet; correctness and descriptor-backed layout
  come first for this dense gate.

## Architecture

Extend the separate TP/EP full-layer smoke rather than the PP appliance:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
  existing Sprint 235 scaffold
  + --dense-compute-tensor blk.2.attn_q_a.weight
  + F8 block-128 row shard loader
  + deterministic [slots x 4096] input
  + f8_dequant_dot_kernel -> [slots x 128] output per GPU
  + CPU oracle sample and repeat check
```

The kernel may compute in FP32 for this gate. That is acceptable because V100
has no native BF16/FP8/FP4 tensor-core math; the important Sprint 236 contract
is that packed F8 bytes stay packed in memory and are expanded inside the GPU
kernel, not pre-expanded into a persistent FP16/FP32 weight arena.

## Implementation

1. Add an opt-in `--dense-compute-tensor NAME` argument to
   `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
2. Locate exactly eight `dense_tp` rows for the selected tensor and layer.
3. Validate that:
   - source dtype is `f8_e4m3_b128`;
   - source shape is `[4096x1024]`;
   - `shard_count=8`;
   - each rank owns `128` rows;
   - `bytes_estimate=528384` per rank.
4. Load each rank's physical F8 shard from the production sidecar.
5. Allocate per-GPU input/output buffers for `32` slots.
6. Implement a CUDA kernel that computes:

   ```text
   out[slot, local_row] =
     dot(dequant_f8_b128(weight[local_row, :]), input[slot, :])
   ```

7. Run warmup and timed repeat iterations.
8. Verify:
   - output values are finite;
   - repeat output is deterministic;
   - CPU oracle sample agrees within tolerance.
9. Keep existing Sprint 235 summary fields and add:
   - dense tensor name;
   - dense compute rows/cols/slots;
   - dense compute bytes loaded;
   - dense compute ms;
   - dense compute repeat max_abs;
   - dense compute oracle max_abs;
   - dense compute pass/fail.
10. Build and run on the V100 pod.
11. Copy evidence to
    `logs/from-cluster/sprint236-tp-dense-compute-gate/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | real dense compute gate |
| `docs/sprints/SPRINT-236.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint236-tp-dense-compute-gate/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Dense compute gate is implemented only in the separate TP/EP tool.
- [ ] No PP scheduler files are modified.
- [ ] Tool resolves `blk.2.attn_q_a.weight` from the real TP/EP contract.
- [ ] Tool loads packed F8 shards directly from the production pack.
- [ ] Kernel expands F8 values inside the GPU computation path.
- [ ] V100 run passes finite repeat checks for the dense output.
- [ ] V100 run passes bounded CPU oracle comparison with declared tolerance.
- [ ] Existing full-layer scaffold, KV, and EP checks still pass.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint236-tp-dense-compute-gate/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- This first dense gate uses FP32 accumulation, not HMMA/CUTLASS. It proves
  descriptor-backed packed-byte computation, not final performance.
- CPU oracle tolerance must account for different accumulation order.
- If this kernel is much slower than checksum scaffolding, that is expected;
  the next optimization step is a fused FP16/HMMA or CUTLASS-backed version
  once correctness and layout are proven.
- `blk.2.attn_q_a.weight` is one dense tensor, not the whole layer.

## Decision

Pending.
