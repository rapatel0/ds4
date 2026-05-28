# TEMP_STATUS_REPORT_038

Date: 2026-05-24

## Topline

Sprint 326 moved the TP/EP compressed-attention diagnostic from one visible
compressed row to bounded multi-row history.

Implemented in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

- `kBoundedCompRows = 8`
- layer-local counters for attention compressed rows and indexer compressed rows
- ring-row append for emitted compressed rows
- bounded multi-row ratio-4 indexer scoring
- top-k index replication from rank 0 to the other TP ranks
- raw+compressed attention over multiple selected compressed rows
- compact reference diff against live pre-shift compressor state

## V100 Validation

Build:

- Command: `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- Result: PASS

Passing gate:

- `slots=32`
- `ctx=262144`
- `position=262135`
- `decode_steps=8`
- `layers=43`
- `pass_invocations=344`
- `projected_slot_step_tok_s=20.780883`
- `checksum=2118198918`
- Result: PASS

Evidence:

- `visible_compressed_rows=2` appears for ratio-4 layers.
- `selected_compressed_rows=2` appears in raw+compressed attention reads.
- `grep -E 'DIFF|FAIL'` on the passing log returns no rows.

## Interpretation

The TP/EP smoke path now proves bounded multi-row compressed history and
multi-row attention selection at the target `32` slot / `256K` shape.

This is still not the final production KV allocator. It keeps only `8` bounded
rows and remains a diagnostic cache. The next gap is production compressed-KV
ownership and a stronger full-reference attention-output comparison.

## Artifacts

- `logs/from-cluster/sprint326-bounded-multirow/cluster/alllayers-slots32-pos262135-steps8-attn-only-v3.log`
