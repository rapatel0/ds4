# Sprint 579 - C1 Full-Capture Batch-Instability Fixed

Date: 2026-05-29

## Goal

Pin the exact source of the full-capture batch-instability (Sprints 576-578) with
a runtime trace and fix it.

## Result summary

**Fixed.** Runtime per-stage differential tracing localized the divergence to the
captured `compressed_kv` stage's attention dense output (`attn_q_b.d_out`), caused
by an **asymmetric stream barrier** that let `dense_stream` lap `rank_stream`
across graph replays. Making the barrier bidirectional (matching the eager
two-stream drain it substitutes for) eliminated the instability: two identical
full-capture runs went from diverging on every slot to **bit-identical**.

## Runtime localization

Static analysis (Sprint 578) was exhausted, so used per-stage replay checksums
(`--decode-stage-checksum-gate` -> `log_replay_stage_checksums`) to diff two
full-capture runs at **matched positions** (achieved with a leading dummy leg).
Result: tokens 0-2 bit-identical; **token 3 first diverges at layer 19, stage
`compressed_kv`, tensor `attn_dense_out`** (then layers 20+). `hc_current` and
`attention_projection` (earlier in the pipeline) matched, so the divergence
originates in `compressed_kv`. Token 3 = position `250003` = the first ratio-4
compressed-KV emit position, so it is emit-triggered.

`attn_dense_out` = `attn_q_b.d_out`, which `run_true_ds4_compressed_kv_projection_gate`
modifies in place on `dense_stream` (`head_rms_norm_local_heads_kernel` +
`rope_tail_rows_kernel`, `compressed_kv_step.cu:1493-1508`).

## Root cause

At the end of that dense block (`compressed_kv_step.cu:1537-1548`), the eager path
drains **both** streams (`cudaStreamSynchronize` on `dense_stream` and `stream`),
but the captured path substitutes `enqueue_rank_streams_wait_after_dense_streams`
(`engine/output_head.cu:1333`), which was **dense->rank only**: it recorded an
event on `dense_stream` and made `rank_stream` wait, but never the reverse. With
only that one-directional edge, `dense_stream` could lap `rank_stream` across
graph replays, racing the in-place writes to `attn_q_b.d_out` against the prior
step's readers. This is graph-replay-specific (eager fully drains), scales with
active routed tokens (more `rank_stream` work to lap), and surfaces at emit
positions -- matching every observation from Sprints 576-578.

## Fix

Made `enqueue_rank_streams_wait_after_dense_streams` a **bidirectional barrier**:
in addition to `rank_stream` waiting on the `dense_stream` event, also record a
`rank_stream` event and make `dense_stream` wait on it. This restores the two-way
ordering the eager `cudaStreamSynchronize` pair implied. The change only adds
event ordering -- it cannot alter computed values, so it is correctness-preserving
by construction.

## Validation

Rebuilt the appliance (`BUILD_EXIT=0`). Full-vs-full logit/sequence floor at
`SLOTS=8`, matched positions (short prompt so all 8 requests coalesce into one
batch at `250000`):

| metric | before fix | after fix |
| --- | ---: | ---: |
| `full-A` vs `full-B` generated-sequence mismatch | `8/8` (Δ up to `3.63`) | **`0/8`** |
| first-diff offsets | all `0` | none |

Two identical full-capture runs are now bit-identical. The remaining within-batch
slot variation (7 distinct sequences for 8 identical short prompts) is reproduced
identically across runs -- the same benign batch-reduction-order behavior eager
exhibits; the run-to-run nondeterminism that was the defect is gone.

## Decision

The C1 full-capture batch-instability defect is fixed. The fix lands in the
shared captured-region ordering helper (`output_head.cu`), so it benefits every
cudagraph path (suffix-replay and full-capture).

Next: a standard serving parity/perf promotion gate for no-suffix full capture
against the eager floor (now that it is deterministic), and a re-confirm of the
promoted suffix-control path under the strengthened barrier. Promotion of the
launcher default remains a separate, gated decision.

## Definition of Done

- Runtime per-stage localization recorded (compressed_kv / `attn_q_b.d_out`).
- Root cause (asymmetric dense<->rank barrier) and fix (bidirectional barrier)
  recorded.
- Validation recorded: full-vs-full mismatch `8 -> 0`.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.
