---
sprint: 325
title: TP/EP Layer-Local Compressed Attention Reference Diff
status: completed
started: 2026-05-24
completed: 2026-05-24
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

- [x] CUDA build passes on the V100 pod.
- [x] Layer-2 ratio-4 diagnostic runs at `slots=1` and `position=100003`.
- [x] Layer-2 ratio-4 diagnostic runs at `slots=32` and `position=262143`.
- [x] The diagnostic prints diff summaries for all target tensors.
- [x] The sprint records the first divergent tensor and proposes the next fix.
- [x] `VISION.md`, `TEMP_STATUS_REPORT_037.md`, cluster artifacts, and this
  sprint doc are updated and committed.

## Outcome

Implemented `--true-ds4-compressed-reference-diff-gate` in the TP/EP
full-layer smoke. The gate compares compact same-kernel reference-layout
tensors for ratio-4 compressed attention:

- `attn_comp_kv_current_peer_copy`
- `attn_comp_score_current_peer_copy`
- `attn_comp_row0_compact_reference`
- `index_comp_row0_compact_reference`
- `indexer_score_row0_compact_reference`

This is intentionally a compact diagnostic reference. It is not yet a full
non-TP `ds4_v100_layer_execute` reference replay, and it only validates the
currently bounded one-row compressed-cache path.

The first all-layer attempt found a concrete bug: layer `2` passed, but layer
`4` diverged at `attn_comp_row0_compact_reference`. The root cause was that the
smoke reused raw-SWA, attention-compressed, and indexer-compressed state across
layers. The fix gives each layer its own raw/compressed/indexer state buffers
and aliases the active layer into the existing execution code.

After the fix:

- `slots=1`, `position=100003`: all `43` layers passed; ratio-4 compact
  reference diffs pass through layer `42`.
- `slots=32`, `position=262143`: all `43` layers passed; ratio-4 compact
  reference diffs pass through layer `42`.
- The prior layer-4 `attn_comp_row0_compact_reference` failure now reports
  `max_abs=0` and `PASS`.

The bounded compressed-row cache capacity is now `1` in this diagnostic path.
That matches the currently implemented one-row visible compressed attention
semantics and avoids allocating a full `slots * 1024 * 512 * 43` float cache in
the smoke. A production TP runtime still needs the full compressed-row cache,
row selection, and long-history lifecycle.

## Validation

Build on `llm/llamacpp-build-8gpu`:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS. The only warning is the existing unused
`rms_norm_plain_rows_kernel` warning.

Cluster artifacts:

- `logs/from-cluster/sprint325-compressed-reference-diff-v2/cluster/alllayers-slots1-pos100003.log`
- `logs/from-cluster/sprint325-compressed-reference-diff-v2/cluster/alllayers-slots32-pos262143-row1.log`

Topline diagnostic metrics:

| Case | Result | Projected slot-step tok/s | Checksum |
|---|---:|---:|---:|
| `slots=1`, `position=100003` | PASS | `3.656366` | `4518783943` |
| `slots=32`, `position=262143` | PASS | `39.258626` | `1089553077` |

These are heavily instrumented diagnostic all-layer smoke metrics, not serving
throughput.

## Next Fix

Move from compact same-layout compressed-row diffs to full attention semantics:

- full compressed-row cache beyond one visible row
- ratio-4 / ratio-128 row selection against resident history
- raw+compressed attention output parity against the reference layer path
- HTTP parity rerun once attention output is proven locally

## Risks

- Binding the non-TP reference path inside the TP harness may be awkward because
  the production reference layer executor owns its own cache structures.
- If upstream hidden/q/kv already diverges before compressed attention, the
  sprint should stop there and not overfit the compressed-row code.
- Host readback is acceptable for this diagnostic; it must not leak into the
  production serving path.
