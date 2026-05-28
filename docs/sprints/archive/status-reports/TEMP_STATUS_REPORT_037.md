# TEMP_STATUS_REPORT_037

Date: 2026-05-24

## Topline

Sprint 325 completed the next TP/EP compressed-attention diagnostic. The focus
was not serving speed; it was finding the first concrete tensor divergence in
the bounded compressed-KV/indexer path added by Sprint 324.

Implemented in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

- `--true-ds4-compressed-reference-diff-gate`
- compact diff summaries for ratio-4 compressed attention tensors
- layer-local raw-SWA, attention-compressed, and indexer-compressed state
- a bounded row pack helper for comparing emitted compressed rows

The first all-layer diagnostic found a real bug: layer `2` passed, but layer
`4` diverged at `attn_comp_row0_compact_reference`. The cause was cross-layer
state reuse in the smoke harness. After making the attention/indexer cache
state layer-local, both target cases pass.

## V100 Validation

Build:

- Command: `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS
- Warning: existing unused `rms_norm_plain_rows_kernel`

Runs:

| Case | Result | Projected slot-step tok/s | Checksum |
|---|---:|---:|---:|
| `slots=1`, `position=100003` | PASS | `3.656366` | `4518783943` |
| `slots=32`, `position=262143` | PASS | `39.258626` | `1089553077` |

Layer-2 `slots=32` target tensors all pass with `max_abs=0`:

- `attn_comp_kv_current_peer_copy`
- `attn_comp_score_current_peer_copy`
- `attn_comp_row0_compact_reference`
- `index_comp_row0_compact_reference`
- `indexer_score_row0_compact_reference`

The former layer-4 failure now also passes with `max_abs=0`.

## Interpretation

This confirms the TP/EP smoke path was previously mixing compressed attention
state across layers. That is now fixed in the diagnostic harness.

This does not yet prove full DS4 compressed-attention parity. The reference is
compact and same-layout, and the compressed-row cache remains bounded to one
visible row. The full production path still needs long-history compressed-row
cache allocation, ratio-4 / ratio-128 row selection, and raw+compressed
attention-output parity against the reference layer path.

## Current Gap To Usable TP/EP Serving

The HTTP API is askable, tokenizes text, keeps session state, and can generate
tokens, but it is not model-correct yet. The official reference vector still
needs to match before treating output quality as valid.

Next correctness work:

- full compressed-row cache beyond one diagnostic row
- reference diff for raw+compressed attention output, not only emitted rows
- rerun HTTP parity after local attention parity moves
- then return to serving hardening and throughput optimization

MTP remains deferred until TP/EP model correctness and serving behavior are
stable.

## Artifacts

- `logs/from-cluster/sprint325-compressed-reference-diff-v2/cluster/alllayers-slots1-pos100003.log`
- `logs/from-cluster/sprint325-compressed-reference-diff-v2/cluster/alllayers-slots32-pos262143-row1.log`
