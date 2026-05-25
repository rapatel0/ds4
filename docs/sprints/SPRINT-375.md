# Sprint 375: Async Output Sync Removal

## Overview

Implement the first throughput-pivot gate from `TEMP_THROUGHPUT_PROMPT.md`:
`--async-output-gate`.

The purpose is not to optimize sampling by itself. The purpose is to remove
host synchronization from the steady-state decode step so Sprint 376 can
attempt CUDA graph capture. Sprint 371 showed full 32-slot decode is still
around `98` aggregate server decode tok/s with about `10%` average GPU
utilization, flat across active request counts. That points to launch/sync
fragmentation rather than raw compute saturation.

## Scope

- Add a default-off CLI gate:

```text
--async-output-gate
```

- Audit steady-state decode in
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, especially the token-major
  `run_one_step` region.
- Move selected-token/output-head D2H synchronization out of the steady-state
  region where possible.
- Use CUDA stream/event sequencing for selected-token copy readiness.
- Preserve current default behavior when the gate is off.
- Validate on the V100 pod with same-binary A/B.

## Out Of Scope

- Do not implement CUDA graph capture in this sprint.
- Do not change output-head math, token selection, tokenizer behavior, or
  session semantics.
- Do not modify PP/layer-split code.
- Do not promote the gate unless the V100 A/B proves it safe and useful.

## Implementation Notes

The throughput prompt identified the current problem shape:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu:
  many cudaDeviceSynchronize/cudaStreamSynchronize calls
  zero CUDA graph calls
  token-major run_one_step is the capture target
```

The implementation should:

1. Add an `async_output_gate` boolean to the existing CLI/options flow.
2. Add per-rank or output-rank CUDA stream/event resources for selected-token
   copy readiness.
3. Replace hot-step host syncs around token copy/output consumption with
   stream-ordered events.
4. Synchronize only at the point where the CPU genuinely needs the selected
   token for the next step's embed seed or for final reporting.
5. Keep all diagnostic checksum/parity paths intact.

If a synchronization call is required for correctness and cannot be moved,
record it explicitly in the sprint result. Sprint 376 should not begin until
the remaining graph blockers are known.

## Validation

Build on the V100 pod:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run same-binary A/B at the real shape:

```text
tools/ds4-v100-tp-ep-active-slot-matrix.py
tools/ds4-v100-tp-ep-http-ab.py
```

Required reported fields:

- active requests
- configured slots
- context
- generated/decode tok/s
- average and max GPU utilization
- first token
- all-layer decode checksum
- sync-count audit before/after, or the list of remaining steady-state syncs

## Definition Of Done

- `--async-output-gate` builds and defaults off.
- V100 build passes.
- Same-binary A/B artifacts are copied to
  `logs/from-cluster/sprint375-async-output`.
- First token and decode checksum are unchanged.
- GPU utilization or server decode tok/s is flat-or-up.
- Sprint doc records an explicit PROMOTE or REJECT decision.
- `docs/sprints/VISION.md` and `docs/sprints/STATUS.md` are updated.
- Changes are committed.

## Decision Rule

Promote only if the gate preserves first token/checksum and improves either
GPU utilization or server decode tok/s. If it is flat but removes the final
CUDA graph blocker, keep it opt-in and carry the evidence into Sprint 376. If
it changes tokens/checksum, reject it.
