# Sprint 370: TP/EP Active-Slot Utilization Matrix

## Overview

Add a reusable active-slot matrix driver for the TP/EP profile harness.

Sprint 369 made utilization visible for a single profile case. Sprint 370 makes
that measurement repeatable across active request counts so scheduling work can
be driven by evidence instead of one-off observation.

## Scope

- Add a permanent matrix tool that runs
  `tools/ds4-v100-tp-ep-profile.py` across request-count cases.
- Default matrix: `1,4,8,16,32` active HTTP requests against `32` configured
  TP/EP slots and `256K` context.
- Preserve the same profile artifacts for every case.
- Write aggregate `active_slot_matrix.tsv` and `active_slot_matrix.json`.
- Keep profiling/sampling controls explicit and disabled unless requested by
  the caller.

## Out Of Scope

- No PP/layer-split work.
- No kernel changes.
- No MTP work.
- No active-slot scheduler optimization yet. This sprint produces the harness
  needed to select and validate that optimization.

## Definition Of Done

- Matrix tool has `--help` and validates request-count inputs.
- Local syntax checks pass.
- A small V100 matrix run succeeds and writes TSV/JSON summaries.
- The result records throughput, coalesced batch size, GPU utilization, and
  per-case artifact paths.
- Sprint docs/status are updated and committed.

## Outcome

Implemented `tools/ds4-v100-tp-ep-active-slot-matrix.py`.

Validation:

- `python3 -m py_compile tools/ds4-v100-tp-ep-active-slot-matrix.py
  tools/ds4-v100-tp-ep-profile.py`: pass locally and on the V100 pod.
- `tools/ds4-v100-tp-ep-active-slot-matrix.py --help`: pass.
- V100 smoke matrix:
  - shape: `32` configured slots, `256K` context, `position=100000`,
    `2` tokens/request, active request cases `1,4`
  - case `1`: `1/1` HTTP 200, `coalesced_batch_size=1`,
    server decode `101.842964` tok/s, average GPU util `8.341667%`,
    max GPU util `41%`
  - case `4`: `4/4` HTTP 200, `coalesced_batch_size=4`,
    server decode `101.159316` tok/s, average GPU util `8.333333%`,
    max GPU util `39%`

Artifacts:

- Cluster:
  `/workspace/logs/sprint370-active-slot-matrix-smoke`
- Local:
  `logs/from-cluster/sprint370-active-slot-matrix-smoke`

Interpretation:

The matrix driver works and writes aggregate TSV/JSON plus per-case profile
artifacts. The two-case smoke is not a full performance characterization, but
it confirms coalescing is active while server-side decode and GPU utilization
remain essentially flat from `1` to `4` active requests at this short shape.
The next measurement should run the full `1,4,8,16,32` matrix with longer
decode, then use that result to choose active-slot compaction versus deeper
dense/state kernel work.
