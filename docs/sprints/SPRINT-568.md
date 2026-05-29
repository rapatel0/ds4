# Sprint 568 - C1 Full-Capture HC Buffer Rebase

Date: 2026-05-29

## Goal

Repair the inter-layer current/HC pointer-buffer mismatch that blocks
no-suffix full-capture cross-position replay.

## Context

Sprints 566 and 567 cleared two misleading diagnostics: rank-major
`current_full` drift was a timing artifact, and full-buffer `route_a` drift was
scratch noise. The remaining graph-vs-graph signal is layer-1 current drift
after a prior layer replay whose end-of-step output tensors still looked clean.

The likely cause is final-HC ping-pong state. CUDA graph capture records fixed
device pointer arguments for `d_final_hc_shard` and `d_hc_scratch_shard`, while
eager execution advances the logical HC state by swapping those host pointers.
After a cache-hit replay, the host swap is mirrored, but the next replay may
launch a graph whose captured input pointer is no longer the current logical HC
buffer.

## Plan

1. Store the captured full-capture final-HC input and output buffer addresses
   per rank.
2. On no-suffix full-capture cache hit, rebase live current HC contents into
   the captured input buffer before launching the graph.
3. Set host pointers to the captured input/output pair before launch, so the
   existing post-replay swap points at the captured output buffer.
4. Re-run the temporary relaxed-position diagnostic. Promote only if the
   comparable graph-vs-graph snapshots and selected-token checksums match.

## Definition of Done

- Remote V100 container build passes from a clean rsync.
- Position-keyed graph behavior remains clean.
- Temporary relaxed-position full-capture replay matches the three-request
  selected-token/checksum probe or produces a later, cleaner first-diff.

## Result

Implemented full-capture HC buffer rebasing:

- `TpCudaGraphLayerExec` now records the captured final-HC input/output buffer
  addresses per rank.
- Before a no-suffix full-capture cache-hit replay, live logical HC contents are
  copied into the captured input buffer if the host pointer has advanced to the
  alternate ping-pong buffer.
- Host pointers are reset to the captured input/output pair before graph launch,
  so the existing post-replay host swap points at the graph's captured output
  buffer.
- No-suffix full-capture cache hits no longer require the position key or
  current final-HC pointer key when this rebase path is available. The graph
  option remains opt-in; this does not make full capture a serving default.

Validation:

- Remote production build passed in `/workspace/s568-hc-rebase-promoted`.
- A temporary graph-vs-graph comparison first validated the repair before
  promotion:
  - position-keyed graph control: `/workspace/s568-hc-rebase-control`
  - relaxed graph candidate: `/workspace/s568-hc-rebase`
  - artifacts: `/workspace/s568-graph-compare-artifacts`
  - result: three-request selected-token/checksum match, including the old
    failing third request.
- The stronger six-request graph-vs-graph check passed:
  `/workspace/s568-graph-compare-6-artifacts`.
- The promoted tree then passed a six-request eager-vs-full-graph probe with no
  remote source patch:
  `/workspace/s568-promoted-6-artifacts`.

Promoted six-request selected tokens/checksums:

- `24426` / `128829740021`
- `2039` / `106648190597`
- `117465` / `17092309830`
- `61356` / `19814694371`
- `25681` / `110664132098`
- `115959` / `18767001400`

The promoted full-graph leg reported `43` captures, `215` persistent cache-hit
replays, and zero persistent invalidations across the six requests.

## Decision

Promote the HC buffer rebase and no-suffix full-capture position-key relaxation
for the opt-in full-capture replay path. This is a correctness/readiness repair,
not a serving-default performance promotion.

The next C1 work should move from reduced selected-token correctness to serving
parity/performance metrology: controlled stochastic settings, warmup before
measurement, startup/init time excluded, and a longer generation prompt before
any throughput claim.
