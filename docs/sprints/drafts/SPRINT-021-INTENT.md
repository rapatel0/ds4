# SPRINT-021 Intent: Executor-Owned Compressor/Indexer Decode Rows

## Intent

Move DS4 ratio-layer compressed-row generation from test fixtures into the
V100 layer executor.

Sprint 020 proved descriptor ownership and HC scheduling. Sprint 021 should make
the layer path generate attention compressed rows, ratio-4 indexer compressed
rows, and ratio-4 compressed-row visibility from real source descriptors.

## Success Shape

- Layer-2 executor accepts mutable decode-cache state.
- The executor projects attention compressor KV/score rows from `attn_norm`.
- The executor updates attention compressor recurrence and appends emitted
  compressed attention rows.
- The executor projects ratio-4 indexer compressor KV/score rows from
  `attn_norm`, updates indexer recurrence, and appends emitted indexer rows.
- When `n_comp > 512`, the executor scores indexer rows, emits top-k indices,
  and calls indexed mixed attention.
- V100 validation proves at least one descriptor-bound emitted row and one
  integrated layer pass through executor-owned compressed state.

## Stop Condition

Stop if the existing CUDA compressor/indexer helpers cannot represent the
source-layout DS4 recurrence without changing cache/state layout.
