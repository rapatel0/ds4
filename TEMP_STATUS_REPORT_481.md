# TEMP Status Report 481

Date: 2026-05-28

## Sprint 481 - Code Cleanup

Snapshot baseline:

- `e65614cb` - `Pre-cleanup snapshot: state before TEMP_CODE_CLEANUP_PROMPT`
- Tag: `pre-cleanup-snapshot`
- Branch pushed before cleanup: `origin/claude-takeover`

Cleanup commits landed:

- `14c773f2` - archived numbered `TEMP_STATUS_REPORT_001.md` through
  `TEMP_STATUS_REPORT_475.md` to `docs/sprints/archive/status-reports/`.
- `d01917a6` - archived superseded root `TEMP_*.md` topic docs to
  `docs/sprints/archive/`.
- `df9250e8` - removed the retired `--decode-cudagraph-peer-copy-gate`
  parser/option/status plumbing from `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

## A6 Status

A6 PATH 4 was attempted first and then backed out.

Validation evidence:

- Control baseline completed `256/256` HTTP 200 at the reference shape using an
  in-pod load generator after local `kubectl port-forward` proved unstable.
- Control status reported `peer_copy_sys_bytes=8`, so the full zero-SYS gate did
  not clear in that run.
- Candidate with the A6 rank-major norm revive became ready but returned
  `0/256` HTTP 200 generation responses. It is not promoted.
- The A6 CUDA diff was removed from the worktree before cleanup continued.

Follow-up: retry A6 from a clean branch/sprint with a narrower harness and
server-side failure capture before touching the cleanup baseline again.

## Repo Cleanup

Root status reports:

- Before: `189` numbered root `TEMP_STATUS_REPORT_*.md` files.
- After: rolling active window only. `TEMP_STATUS_REPORT_477.md` through
  `TEMP_STATUS_REPORT_481.md` remain at root.

Superseded root topic docs archived:

- `TEMP_CURRENT_REPORT.md`
- `TEMP_GRAPH_PRIOR_INSIGHTS.md`
- `TEMP_HC_ALLREDUCE_PROMPT.md`
- `TEMP_HC_ALLREDUCE_STEER.md`
- `TEMP_NCCL_BROADCAST_REDUCTION_AUDIT.md`
- `TEMP_SPIKE_A_VLLM_PORT.md`
- `TEMP_SPIKE_B_C_CAPTURE.md`
- `TEMP_STATUS_REPORT.md`
- `TEMP_SYS_TRANSPORT_SWEEP.md`
- `TEMP_THROUGHPUT_PROMPT.md`

`tools/ds4-source-oracle-vector.{c,o}` was audited and retained. It is not an
orphan: `Makefile` still builds `tools/ds4-source-oracle-vector`, and
`tools/ds4-v100-gate.sh` still runs it as the `source_guards` gate.

## Code Cleanup

Removed:

- `Options::decode_cudagraph_peer_copy_gate`
- `--decode-cudagraph-peer-copy-gate` parse branch
- retired-flag rejection block
- token-major scaffold field `decode_cudagraph_peer_copy`

Active-file grep now finds no `decode_cudagraph_peer_copy` references.

Transport wrapper audit:

- `ds4_peer_copy_async` is absent from `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- `enqueue_graph_f32_copy_from_device0` and
  `enqueue_graph_f32_copy_between_devices` remain reachable in
  `decode_cudagraph_gate` graph branches. They were not deleted in this pass.

## Validation

- `git diff --check -- tools/ds4-v100-tp-ep-full-layer-smoke.cu`: pass.
- V100 build in `llm/ds4-tp-bench:/workspace/ds4-sprint181`: pass.

Build command:

```bash
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Only pre-existing unused-kernel warnings were emitted for
`rms_norm_plain_rows_kernel` and `indexer_score_row0_slots_kernel`.
