# TEMP Status Report 082

Date: 2026-05-25

## Current Focus

TP/EP active-slot metrology. Sprint 370 added a reusable matrix driver around
the permanent TP/EP profile harness so active request counts can be compared
with identical artifacts.

## New Tool

`tools/ds4-v100-tp-ep-active-slot-matrix.py`

Default intent:

- Run active request cases `1,4,8,16,32`
- Keep configured slots at `32`
- Keep context at `256K`
- Use the existing profile harness for each case
- Write aggregate:
  - `active_slot_matrix.tsv`
  - `active_slot_matrix.json`
- Preserve each case's full profile artifact directory

## Validation

Local:

- `python3 -m py_compile tools/ds4-v100-tp-ep-active-slot-matrix.py
  tools/ds4-v100-tp-ep-profile.py`: pass
- `tools/ds4-v100-tp-ep-active-slot-matrix.py --help`: pass
- `git diff --check`: pass before V100 run

V100 smoke matrix:

Command shape:

```text
--requests-cases 1,4
--tokens 2
--ctx 262144
--slots 32
--position 100000
--gpu-sample-interval-ms 200
--hc-current-stream-sync
--tool none
```

Results:

| Active requests | HTTP 200 | Coalesced batch | Server decode tok/s | Avg GPU util | Max GPU util |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1 | 101.842964 | 8.341667% | 41% |
| 4 | 4/4 | 4 | 101.159316 | 8.333333% | 39% |

Artifacts:

- Cluster: `/workspace/logs/sprint370-active-slot-matrix-smoke`
- Local: `logs/from-cluster/sprint370-active-slot-matrix-smoke`

## Interpretation

The matrix tool is operational. The two-case smoke is deliberately short and is
not the final throughput characterization, but it shows the shape of the
problem: coalescing is happening, while server-side decode and GPU utilization
do not improve from 1 to 4 active requests. The next useful run is the full
`1,4,8,16,32` matrix with longer decode, then an implementation sprint based
on that evidence.
