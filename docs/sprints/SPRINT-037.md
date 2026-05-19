# Sprint 037: Resident MTP Raw Attention

## Status

Complete.

## Overview

Sprint 037 adds the next readiness rung toward `missing=mtp_forward`: resident
MTP raw/SWA attention on gpu7. Sprint 036 proved the FFN half of the MTP block
from sidecar-resident bytes; this sprint proves the attention half can consume
sidecar-resident sinks, update the separate MTP raw cache, and produce heads
that match a CPU reference over the same raw-cache ring.

This sprint is intentionally narrow. It does not clear `mtp_forward`; it creates
the resident attention primitive and gate evidence needed before MTP logits,
draft verification, and replay rollback can be integrated.

## Use Cases

- Validate that the MTP sidecar can run raw/SWA attention without falling back to
  mmap-resolved weight pointers.
- Exercise MTP raw-cache row update and wrap visibility (`pos % 128`) with
  native `il=1` semantics.
- Add a gate rung that narrows readiness from generic MTP forward work to the
  remaining logits/verify/rollback work.

## Architecture

- Keep `docs/architecture/DS4-V100-LAYOUT.md` as the topology and residency
  anchor: MTP remains a gpu7 sidecar arena, not a stage-pack tensor.
- Add an arena-backed CUDA attention wrapper that mirrors
  `ds4_gpu_attention_decode_heads_tensor` but resolves `attn_sinks.weight` from
  a `ds4_gpu_arena` view.
- Add `tools/ds4-v100-mtp-attn-smoke.c` as a focused CUDA smoke:
  - upload/open the real MTP sidecar on gpu7;
  - bind `mtp.0.attn_sinks.weight`;
  - allocate synthetic `q`, `kv`, `raw_cache`, and `heads`;
  - write KV rows through the production raw-cache store primitive;
  - decode attention for positions including wrap cases;
  - compare against a CPU sink-aware reference.

## Implementation

1. Extend `ds4_gpu.h`, `ds4_cuda.cu`, and `ds4_gpu_arena_stub.c` with
   `ds4_gpu_arena_attention_decode_heads_tensor`.
2. Add the MTP attention smoke tool and wire it into `Makefile`.
3. Add the `mtp_attn` gate to `tools/ds4-v100-gate.sh`.
4. Update readiness ordering so `mtp_attn` must pass before the remaining
   `mtp_forward` blocker.
5. Write the sprint report and refresh `docs/sprints/VISION.md`.

## Files Summary

- `ds4_gpu.h`
- `ds4_cuda.cu`
- `ds4_gpu_arena_stub.c`
- `tools/ds4-v100-mtp-attn-smoke.c`
- `tools/ds4-v100-gate.sh`
- `Makefile`
- `docs/sprints/SPRINT-037-REPORT.md`
- `docs/sprints/VISION.md`

## Definition of Done

- Local compile checks pass for the new tool object and changed CPU stubs.
- CUDA build on the V100 cluster passes for
  `tools/ds4-v100-mtp-attn-smoke`.
- Focused MTP attention smoke passes on gpu7 with all 8 V100s visible.
- Full V100 gate includes `mtp_attn PASS`, has no failures, and still reports
  `ready=false missing=mtp_forward`.
- Sprint report records commands, outputs, tolerances, and remaining blockers.
- Vision document names Sprint 037 and updates the readiness ladder.

## Results

- Local C/stub build passed:
  `make tools/ds4-v100-mtp-attn-smoke.o ds4_v100_mtp.o ds4_gpu_arena_stub.o ds4_cpu.o`
- Local shell/diff checks passed:
  `bash -n tools/ds4-v100-gate.sh`, `git diff --check`
- Cluster CUDA build passed:
  `CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-attn-smoke`
- Focused cluster smoke passed:
  `mtp_attn_smoke PASS`
- Full 8-GPU gate passed:
  `gate summary PASS failures=0 ready=false`
- Readiness remains intentionally blocked on:
  `missing=mtp_forward`

## Risks

- The focused smoke uses synthetic Q/KV to isolate raw-cache semantics; full
  MTP attention projection and grouped output remain future work.
- Attention tolerance must account for CUDA reduction ordering but should stay
  tight because the smoke uses the same F32 raw rows read back from device.
- The arena wrapper must not create a second path with diverging semantics from
  the mmap-backed attention wrapper.

## Security

No network-serving surface changes. The sprint only adds local CUDA smoke
coverage and a new arena-backed device primitive.

## Dependencies

- Sprint 031 sidecar arena residency.
- Sprint 034 prefix chain for the later full MTP input.
- Sprint 036 FFN slice and readiness gate.
- Real MTP sidecar model at the cluster path used by prior gates.

## Open Questions

- Whether Sprint 038 should fold projections and grouped attention output into
  this smoke before logits, or go directly to an integrated one-token MTP block.
- Whether replay snapshot/rollback should begin before logits parity or after a
  complete draft token can be produced.
