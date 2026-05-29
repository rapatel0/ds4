# Sprint 560 - Full-Capture Emitted Row Host Metadata Mirror

Date: 2026-05-29

## Goal

Advance C1 emitted topology and row-position metadata by removing the next
known replay-time host-state mismatch for no-suffix full-capture cache hits.

## Context

Sprint 559 fixed final-HC pointer metadata after no-suffix full-capture replay.
The next ordered steering item is emitted topology and row-position metadata.
The remaining host-selected state is in the compressed-KV emitted row path:

- `attn_comp_rows_written_layers[layer]`
- `attn_comp_row_position_layers[layer][row]`
- `attn_comp_row_loaded_layers[layer][row]`
- `attn_comp_row_loaded_position_layers[layer][row]`
- equivalent indexer fields for ratio-4 layers

The CUDA graph replays the emitted row kernels, but host counters and row
metadata are not part of graph replay. Since full capture is still position
keyed, this sprint mirrors the eager host metadata update after a successful
same-position cache-hit replay. It does not remove the position key and does
not promote no-suffix full capture as a serving default.

## Plan

1. Add a no-suffix full-capture replay helper that mirrors eager emitted-row
   host metadata for the current layer when `emitted` is true.
2. Keep it guarded to the same graph-safe diagnostic surface:
   - no suffix stage
   - full-capture persistent replay succeeded
   - compressed-KV gate active
   - current position is emitted for the layer ratio
3. Mirror attention compressed-row metadata for all compressed layers.
4. Mirror indexer compressed-row metadata only for ratio-4 indexer layers.
5. Validate at a ratio-4 emitted position so both attention and indexer metadata
   paths run.

## Definition of Done

- Remote build passes.
- A reduced ratio-4 emitted-position served-default control confirms the served
  appliance still keeps compressed KV disabled by default.
- A one-off compressed-KV appliance entrypoint exercises the branch without
  adding a permanent CLI/env flag.
- Eager and no-suffix full-capture replay probes match selected tokens and
  decode checksums with compressed KV enabled.
- Full-capture replay reports `43/43` cached graph replays, zero invalidations,
  zero peer/SYS transport, and zero NCCL graph SYS edges.
- Promoted suffix sanity remains clean.
- `SPIKE_B_STEERING.md` and `docs/sprints/VISION.md` are updated with the
  result.

## Result

Implemented `apply_full_capture_replay_compressed_kv_host_state()` in
`engine/decode_loop.cu`. After a successful no-suffix persistent full-capture
cache-hit replay, it mirrors the eager compressed-row host metadata updates for
the current emitted position:

- attention compressed-row `position`, `loaded`, `loaded_position`, and
  `rows_written`
- ratio-4 indexer compressed-row equivalents when indexer attention is active

The helper is guarded to no-suffix full capture and `true_ds4_compressed_kv_gate`
so the promoted suffix replay and current served default are unchanged.

Validation:

- Build passed earlier in `/workspace/s560-emitted-metadata`.
- Eager served default at ratio-4 emitted position `262083`:
  `/workspace/s560-emitted-metadata-artifacts/none-s560-eager-ratio4-emitted-seq2/summary.json`
  returned `http_200=2`, first selected token `107875`, and confirmed
  `compressed_kv_layers=0`.
- No-suffix full-capture replay served default:
  `/workspace/s560-emitted-metadata-artifacts-user/none-s560-fullgraph-standard-seq2-serverargs-h396a9fa7/summary.json`
  returned `http_200=2`, `43/43` cached graph replays, zero invalidations, zero
  peer/SYS transport, and zero NCCL graph SYS edges. The cache-hit replay
  response selected token `107875`, matching the eager served-default token for
  the same shape.
- One-off compressed-KV validation binary:
  `/workspace/s560-emitted-metadata/appliance/ds4-v100-tp-ep-appliance-s560-compressed`
  was built from a temporary entrypoint that only forced
  `true_ds4_compressed_kv_gate=true` and `true_ds4_indexer_attention_gate=true`;
  the temporary smoke source was removed before commit.
- Compressed-KV eager:
  `/workspace/s560-emitted-metadata-artifacts-user/none-s560-compressed-eager-seq2/summary.json`
  returned `http_200=2`, `compressed_kv_layers=86`,
  `compressed_kv_emitted_layers=42`, selected tokens `58204`, `109597`, and
  decode checksums `7265791446`, `79399742586`.
- Compressed-KV no-suffix full-capture replay:
  `/workspace/s560-emitted-metadata-artifacts-user/none-s560-compressed-fullgraph-seq2-serverargs-h396a9fa7/summary.json`
  returned the same selected tokens and checksums as compressed-KV eager,
  reported `43/43` cached graph replays, and kept invalidations, peer/SYS
  transport, and NCCL graph SYS edges at zero.
- Claude bug-find review confirmed the helper's mark-loaded boolean matches the
  eager happy-path algebra.

Decision:

Promote this as C1 readiness/correctness repair only. Full capture remains
diagnostic-only and position-keyed. The remaining C1 emitted-topology work is
to make emitted/non-emitted work selection graph-stable without adding a
permanent flag matrix.
