---
sprint: 015
title: V100 Descriptor-Bound FFN Compute Gate
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-015 Report: V100 Descriptor-Bound FFN Compute Gate

## Verdict

`SHIP`

Sprint 015 shipped runtime tensor bindings and the first descriptor-bound
real-byte FFN compute gate. The new CUDA smoke reads layer-2 routed MXFP4 and
shared F8 FFN bytes from the source GGUF using pack-index source offsets,
uploads them into a V100 arena at their real shard offsets, executes routed and
shared FFN slices, and compares the summed output against CPU source-format
references.

The appliance is still not ready for serving. This sprint proves real
descriptor-bound FFN compute, not full layer scheduling, real router scheduling,
attention/residual/norm, selected-token decode, public serving, MTP, or
throughput.

## What Shipped

- Added public `ds4_v100_tensor_binding` metadata in `ds4_v100_context`.
- Added binding lookup helpers for semantic tensor id, layer tensor suffix, and
  output head.
- Added source-shape dimension parsing to runtime bindings.
- Added `tests/v100_layer_binding_smoke.c` for local descriptor materialization.
- Added `tests/cuda_v100_descriptor_bound_ffn_smoke.c` for real-byte V100 FFN
  compute from pack descriptors.
- Extended `tools/ds4-v100-gate.sh` to build/run layer binding and
  descriptor-bound FFN checks when `--pack-index` is supplied.

## Evidence

Local validation:

- `make tests/v100_layer_binding_smoke`
- `./tests/v100_layer_binding_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --layer 2`
- `make tests/cuda_v100_descriptor_bound_ffn_smoke.o`
- `bash -n tools/ds4-v100-gate.sh`
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-015-LAYER-BINDING-CLUSTER.log`
  - `v100_layer_binding_smoke: layer=2 gpu=0 routed_expert_bytes=4456448 shared_row_bytes=4128 ok`
- `docs/sprints/drafts/SPRINT-015-DESCRIPTOR-BOUND-FFN-CLUSTER.log`
  - `cuda_v100_descriptor_bound_ffn_smoke: layer=2 expert=0 gpu=0 arena_bytes=11784434944 hidden=4096 mid=2048 ... ok`
- `docs/sprints/drafts/SPRINT-015-GATE-CLUSTER/gate-summary.log`
  - Source guards passed against `/models/DSv4-Flash-256e-fixed.gguf`.
  - Existing CUDA smokes, layer descriptors, layer bindings, and
    descriptor-bound FFN passed on the 8x V100 pod.
  - Gate summary: `PASS`, `failures=0`, `ready=false`.

## Deviations

- The descriptor-bound FFN smoke uses expert 0 and deterministic synthetic
  activation input. It does not run router logits or hash/bias expert
  selection.
- The smoke validates the FFN body only: routed MXFP4, shared F8, SwiGLU, and
  sum. It does not include attention, residual, norm, HC relay, output head, or
  selected-token decode.
- Arena allocation spans the real layer-2 stage offset range
  (`11784434944` bytes). This is acceptable on 32 GB V100 for the gate, but the
  production scheduler should reuse the already resident full arena rather than
  allocating a test-only partial arena.

## Handoff

Sprint 016 should convert this bounded FFN proof into a descriptor-bound layer
slice: real router selection, layer-owned activation/state structs, attention
or residual/norm integration, and a path toward selected-token logits.
