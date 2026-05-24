---
sprint: 325
title: TP/EP Layer-Local Compressed Attention Reference Diff
status: planned
started: 2026-05-24
branch: claude-takeover
---

# Sprint 325 - TP/EP Layer-Local Compressed Attention Reference Diff

## Goal

Build a narrow TP/EP-vs-reference diagnostic for one ratio-4 layer so the next
correctness work is driven by tensor diffs instead of top-token drift.

## Why This Sprint

Sprint 324 made the TP/EP compressed-row lifecycle active and stable at the
target `32` slot / `256K` shape, but HTTP parity still returns `mere` instead
of `16`. The new path is structurally live, so the next useful step is to
compare the internal tensors against the non-TP reference implementation at the
first ratio-4 layer where compressed rows are emitted.

Top-token parity is now too indirect. We need to know which tensor diverges:

- emitted attention compressed row
- emitted indexer compressed row
- indexer q/w score
- selected compressed row
- raw+compressed attention output

## Scope

TP/EP only. No PP/layer-split implementation work. No MTP. Keep this as a
diagnostic harness or explicit debug gate; do not make production serving depend
on host-side reference execution.

## Target Case

Start with:

```text
layer: 2
ratio: 4
position: 100003 or 262143
slots: 1 first, then 32
context: 256K allocation shape
```

Layer `2` is the first ratio-4 layer and position `100003` is exercised during
the current `short_reasoning_plain` prompt prefill. Position `262143` is the
clean all-layer smoke boundary where all ratio-4 and ratio-128 layers emit.

## Implementation Plan

1. Add an explicit debug gate to the TP/EP full-layer smoke:

```text
--true-ds4-compressed-reference-diff-gate
```

2. Capture TP/EP tensors after Sprint 324's compressed update:

- attention compressor current KV/gate
- attention compressed state slice
- emitted attention compressed row
- indexer compressor current KV/gate
- emitted indexer compressed row
- indexer q and weight vectors
- one-row indexer score/top-k
- raw+compressed attention head output

3. Run the equivalent non-TP/reference sequence in a narrow harness:

- use existing `ds4_v100_layer_execute.c` / `ds4_cuda.cu` primitives where
  practical
- otherwise add a CPU/reference CUDA helper scoped to this diagnostic
- keep tensor layouts explicit in the log

4. Print bounded diff summaries:

```text
tensor
shape
max_abs
rms
finite_bad
first_bad_index
reference_max
tp_ep_max
PASS/FAIL
```

5. Use the first large divergence to choose the next implementation sprint:

- compressor update mismatch
- indexer score/top-k mismatch
- raw+compressed softmax/read mismatch
- upstream q/kv/hidden mismatch

## Definition of Done

- CUDA build passes on the V100 pod.
- Layer-2 ratio-4 diagnostic runs at `slots=1` and `position=100003`.
- Layer-2 ratio-4 diagnostic runs at `slots=32` and `position=262143`.
- The diagnostic prints diff summaries for all target tensors.
- The sprint records the first divergent tensor and proposes the next fix.
- `VISION.md`, `TEMP_STATUS_REPORT_037.md`, cluster artifacts, and this sprint
  doc are updated and committed.

## Risks

- Binding the non-TP reference path inside the TP harness may be awkward because
  the production reference layer executor owns its own cache structures.
- If upstream hidden/q/kv already diverges before compressed attention, the
  sprint should stop there and not overfit the compressed-row code.
- Host readback is acceptable for this diagnostic; it must not leak into the
  production serving path.
