---
sprint: 014
title: V100 Real Pack-Index Layer Descriptor Gate
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-014 Report: V100 Real Pack-Index Layer Descriptor Gate

## Verdict

`SHIP`

Sprint 014 shipped the first real pack-index descriptor gate for the V100
appliance path. The gate validates the layer-2 descriptor contract across
attention controls, source-F8 attention projections, compressor/indexer
weights, router metadata, routed MXFP4 experts, shared F8 experts, HC controls,
and the global BF16 output head.

This still does not execute a real model layer. It proves that the runtime can
fail closed on the exact source-layout descriptors that the next layer-compute
sprint must consume.

## What Shipped

- Added `tools/ds4-v100-layer-descriptor-gate`, backed by `ds4_pack_open` and
  `ds4_pack_lookup`.
- Added descriptor validation for layer id, owning GPU, source dtype, runtime
  layout, kernel family, shard file, shard offset, and byte length.
- Validated layer 2 as a ratio-4 layer with indexer descriptors and hash-router
  metadata.
- Added `--pack-index FILE` and `--descriptor-layer N` to
  `tools/ds4-v100-gate.sh`.
- Preserved current appliance-gate behavior when no pack index is supplied.

## Evidence

Local validation:

- `bash -n tools/ds4-v100-gate.sh`
- `make tools/ds4-v100-layer-descriptor-gate`
- `./tools/ds4-v100-layer-descriptor-gate --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --layer 2`
- Negative missing-output fixture failed closed.
- `git diff --check`

Cluster validation:

- `docs/sprints/drafts/SPRINT-014-LAYER2-DESCRIPTOR-CLUSTER.log`
  - `descriptor_summary PASS layer=2 expected=35 failures=0 bytes=4655221720`
- `docs/sprints/drafts/SPRINT-014-GATE-CLUSTER/gate-summary.log`
  - Source guards passed against `/models/DSv4-Flash-256e-fixed.gguf`.
  - Source dtype, BF16 probe, 1M single-slot context/KV, compressor bridge,
    prefill KV, HC relay, projection/attention, bounded logits, MXFP4 MoE, and
    layer descriptor gates passed on the 8x V100 pod.
  - Gate summary: `PASS`, `failures=0`, `ready=false`.

Negative validation:

- `docs/sprints/drafts/SPRINT-014-LAYER2-DESCRIPTOR-NEGATIVE-LOCAL.log`
  - Missing `output.weight` produced `descriptor_summary FAIL`.

## Deviations

- The descriptor gate validates layer 2 by default. Ratio-128, later ratio-4,
  and final layer variants should be covered next.
- The descriptor gate is a strict report/check tool. It does not yet materialize
  runtime-owned descriptor structs or launch descriptor-bound compute.
- The appliance gate remains `ready=false` because real pack layer scheduling,
  shared-expert execution in a real layer, real-model selected-token decode,
  public serving, MTP, and throughput benchmarks remain incomplete.

## Handoff

Sprint 015 should turn the validated descriptors into context-owned runtime
bindings and run one descriptor-bound layer or short selected-token slice using
real pack/shard bytes. That sprint should remove the gap between descriptor
validation and compute execution.
