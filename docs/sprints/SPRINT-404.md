# Sprint 404: NCCL VRAM Ledger

Date: 2026-05-26

## Overview

Sprints 400-403 proved that narrow NCCL serving gates are functionally correct
but not memory-admitted at the target `32` slot / `256K` shape. Sprint 403 also
proved that `--fp8-e5m2-kv` does not buy back the missing headroom because the
TP runtime already defaults to FP8 E4M3 block-128 KV.

Sprint 404 makes the resident memory problem explicit and reusable. Instead of
reading raw `tp_ep_vram` rows by hand, add a ledger tool that parses profile
artifacts, reports per-checkpoint/per-GPU free memory, computes allocation
deltas, and quantifies the NCCL reserve deficit per GPU.

## Constraints

- TP/EP only. No PP/layer-split work.
- No serving default changes.
- The tool must work from existing cluster artifacts.
- The output must identify the next memory-reclaim target numerically.

## Implementation

Add:

- `tools/ds4-v100-tp-ep-vram-ledger.py`

The tool parses one or more profile artifact directories and extracts:

- `tp_ep_vram` rows by label and GPU,
- `tp_ep_vram_summary` rows,
- `tp_ep_hc_final_expand_shared` control bytes,
- `tp_ep_diagnostic_output_head_shared` output-head bytes,
- profile `summary.json` values when present.

It writes:

- JSON summary with per-case labels, checkpoint tables, deltas, and deficits,
- Markdown summary for docs/status.

Primary S404 input:

```text
logs/from-cluster/sprint403-nccl-kv-matrix/control/
logs/from-cluster/sprint403-nccl-kv-matrix/hc-nccl/
logs/from-cluster/sprint403-nccl-kv-matrix/fp8-kv-hc-nccl/
```

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-vram-ledger.py
git diff --check
```

Ledger run:

```text
python3 tools/ds4-v100-tp-ep-vram-ledger.py \
  --case control=logs/from-cluster/sprint403-nccl-kv-matrix/control \
  --case hc-nccl=logs/from-cluster/sprint403-nccl-kv-matrix/hc-nccl \
  --case fp8-kv-hc-nccl=logs/from-cluster/sprint403-nccl-kv-matrix/fp8-kv-hc-nccl \
  --threshold-mib 1536 \
  --out-json logs/from-cluster/sprint404-vram-ledger/vram-ledger.json \
  --out-md logs/from-cluster/sprint404-vram-ledger/vram-ledger.md
```

## Results

Local checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-vram-ledger.py
git diff --check
```

Ledger artifacts:

- `logs/from-cluster/sprint404-vram-ledger/vram-ledger.json`
- `logs/from-cluster/sprint404-vram-ledger/vram-ledger.md`

Case summary at `1536 MiB` NCCL reserve:

| Case | Return | First token | Min free | Max deficit | Failing GPUs | Decode tok/s |
|---|---:|---:|---:|---:|---|---:|
| `control` | `0` | `54639` | `1746 MiB` | `0 MiB` | none | `98.1` |
| `hc-nccl` | `14` | n/a | `1114 MiB` | `422 MiB` | `0,1,4,5,6` | n/a |
| `fp8-kv-hc-nccl` | `14` | n/a | `1114 MiB` | `422 MiB` | `0,1,4,5,6` | n/a |

HC-current NCCL per-GPU reserve deficits:

| GPU | Free at `nccl_after_output_head` | Deficit to `1536 MiB` |
|---:|---:|---:|
| 0 | `1114 MiB` | `422 MiB` |
| 1 | `1462 MiB` | `74 MiB` |
| 2 | `1558 MiB` | `0 MiB` |
| 3 | `1554 MiB` | `0 MiB` |
| 4 | `1482 MiB` | `54 MiB` |
| 5 | `1466 MiB` | `70 MiB` |
| 6 | `1514 MiB` | `22 MiB` |
| 7 | `1538 MiB` | `0 MiB` |

Measured resident deltas:

| Allocation checkpoint | GPU impact |
|---|---|
| NCCL rank-buffer/communicator overhead | `+848` to `+944 MiB/GPU` versus non-NCCL rank buffers |
| TP runtime | `+6794 MiB/GPU` |
| Dense ops | `+21128 MiB/GPU` |
| HC controls | `+372 MiB` on GPU0 only |
| Output head | `+134 MiB` on GPU0, `+130 MiB` on GPUs 1-7 |

Metadata:

| Component | Size |
|---|---:|
| HC control tensors | `317.0 MiB` |
| Output weights aggregate | `1010.0 MiB` |
| Output logits aggregate | `15.8 MiB` |

## Interpretation

The failure is not evenly distributed. GPUs 1, 4, 5, and 6 only need the
output-head allocation to move later or shrink. GPU0 also needs the HC-control
residency addressed.

Output-head residency alone cannot fix NCCL admission because GPU0 would still
be short by about `288 MiB`. HC controls alone cannot fix NCCL admission
because GPUs 1, 4, 5, and 6 would still be below reserve. The first plausible
memory sprint is therefore a paired change:

1. lazy/open-on-demand output head, and
2. stream or shrink GPU0 HC-control residency.

That pair is enough on paper: output-head deferral recovers `130-134 MiB` on
all GPUs, while HC-control streaming/shrinking can recover up to `372 MiB` on
GPU0.

## Decision

Promote the VRAM ledger as a permanent analysis tool.

The next NCCL implementation sprint should not target a single small
allocation. It should target the paired memory plan above, then rerun the S403
matrix to prove whether HC-current NCCL becomes admitted at `32` slots /
`256K`.

## Definition of Done

- Ledger tool exists and parses existing artifacts.
- Local checks pass.
- S403 artifacts produce a JSON and Markdown ledger.
- Sprint doc, status, vision, and temporary report are updated with the
  resulting memory targets.
- Commit all kept artifacts explicitly.

## Decision Gate

The next implementation sprint should target the largest resident allocations
that move the failing NCCL case above the `1536 MiB` reserve. If the ledger
shows that a proposed single allocation cannot close the gap, do not spend a
sprint on that allocation alone.
