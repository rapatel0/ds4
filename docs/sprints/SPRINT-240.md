# Sprint 240 - TP/EP Resident Decode Loop Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 239 proved a one-shot TP/EP next-hidden composition for layer `2` at
`32` slots / `256K`. Sprint 240 turns that into a resident repeated decode-loop
gate. The goal is to measure the serving-shaped execution boundary without
pack-byte reloads or per-step allocation, while staying in the separate TP/EP
codepath and keeping MTP off.

This is still not an HTTP server and not logits equivalence. It is the
necessary benchmarkable step before wiring TP/EP into a production serving
loop.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in `--decode-steps N` mode to the TP/EP full-layer smoke.
- Keep all Sprint 239 production byte bindings and next-hidden composition.
- Make the repeated path resident:
  - TurboMind expert weights stay loaded;
  - dense FP8 weights used by composition stay loaded;
  - route-slot and composition buffers stay allocated;
  - per-step loop does not reread pack files.
- Repeat the representative layer-2 step:
  - run routed expert gate/up and down;
  - run resident F8 dense kernels for attention-output and shared-FFN output;
  - reduce EP outputs into hidden shards;
  - peer-copy EP contributions;
  - compose next-hidden shards;
  - optionally update/check the KV slice outside the timing window.
- Report:
  - decode steps;
  - slots;
  - effective tokens processed;
  - total loop ms;
  - ms/step;
  - aggregate slot-step tok/s;
  - per-step EP ms;
  - per-step dense ms;
  - per-step compose/peer ms if separated;
  - finite/repeat/checksum status.
- Run on the V100 pod at `32` slots / `256K`, MTP off.

## Non-Goals

- No PP scheduler edits.
- No changes to `ds4_v100_scheduler.*`.
- No server/API integration.
- No MTP.
- No logits equivalence claim.
- No full attention softmax implementation.
- No final HMMA/CUTLASS dense optimization.
- No attempt to hide a bad result. If the repeated loop is slow because of
  synchronization or scalar dense kernels, record that as the bottleneck.

## Architecture

Extend only:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
  --compose-next-hidden
  --decode-steps N
```

The `--decode-steps` path should be a resident benchmark path, not a wrapper
around the one-shot smoke that reloads tensors. It should prepare resident
state once:

```text
resident setup:
  load TurboMind EP expert tensors
  load F8 dense composition tensors
  allocate route-slot arrays
  allocate EP contribution buffers
  allocate peer-return buffers
  allocate next-hidden shards

timed loop:
  for step in decode_steps:
    run EP gate/up + down
    run resident dense attn_output_b
    run resident dense ffn_down_shexp
    reduce EP route outputs to hidden shards
    peer-copy contributions to destination ranks
    sum contributions and compose next-hidden shard
```

The timing is not expected to be final throughput because attention is still
representative and dense kernels are scalar correctness kernels. It should
nevertheless tell us whether the TP/EP topology is dominated by communication,
sync boundaries, EP kernels, or dense kernels.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Resident decode-loop gate |
| `docs/sprints/SPRINT-240.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint240-tp-ep-resident-decode-loop/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Implementation stays in the separate TP/EP codepath.
- [x] No PP scheduler files are modified.
- [x] `--decode-steps N` builds on the V100 pod.
- [x] The timed loop keeps dense composition weights resident.
- [x] The timed loop keeps EP and composition buffers resident.
- [x] The loop executes at `32` slots / `256K`, MTP off.
- [x] The run reports aggregate slot-step tok/s and ms/step.
- [x] The run reports finite/checksum/repeat status.
- [x] Existing combined dense coverage and Sprint 239 composition still pass.
- [x] Evidence is copied to
      `logs/from-cluster/sprint240-tp-ep-resident-decode-loop/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Risks

- The scalar FP8/BF16 dense kernels will understate final performance.
- The first resident loop may expose synchronization overhead from peer copies
  and stream barriers.
- The current representative attention path is not full DS4 attention.
- If the measured loop is dominated by scalar dense kernels, Sprint 241 should
  target fused HMMA/CUTLASS dense kernels before server integration.

## Decision

Complete. The TP/EP full-layer smoke now has a resident `--decode-steps N`
mode. It loads the two F8 dense composition tensors once, keeps TurboMind EP
weights resident, keeps route-slot and composition buffers allocated, and
repeats the representative layer-2 step without rereading pack bytes.

The V100 run at `32` slots / `256K`, MTP off, `50` resident steps reports:

```text
tp_ep_decode_loop
steps 50
slots 32
slot_steps 1600
total_ms 92.277411
ms_per_step 1.845548
slot_step_tok_s 17339.021356
ep_ms_per_step 0.319095
dense_ms_per_step 0.756244
compose_ms_per_step 0.770121
dense_loaded_bytes 42270720
ep_contribution_bytes 4194304
ep_return_bytes 4194304
checksum 2382924023
finite_bad 0
PASS
```

The same run preserves Sprint 238 and Sprint 239 checks:

```text
dense_compute_pass 1
bf16_compute_pass 1
compose_pass 1
decode_pass 1
kv max_abs 0.000000000
repeat_bad 0
repeat_nan 0
PASS
```

This is a **representative layer-loop metric**, not end-to-end generated
tok/s. It excludes full attention semantics, all layers, logits, sampling, and
server scheduling. The stage split is still useful: the current resident loop
is dominated by scalar dense kernels plus compose/peer synchronization, not by
TurboMind EP kernels alone.

Evidence:

- `logs/from-cluster/sprint240-tp-ep-resident-decode-loop/layer2-resident-decode-loop-32slot-256k-50steps.log`
