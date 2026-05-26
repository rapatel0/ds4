# TEMP Status Report 404: NCCL VRAM Ledger

Date: 2026-05-26

## Current Focus

TP/EP only. No PP/layer-split work.

Sprint 404 added a permanent VRAM ledger so NCCL admission failures are tied
to concrete resident allocation deltas instead of hand inspection.

## What Changed

Added:

- `tools/ds4-v100-tp-ep-vram-ledger.py`

The tool parses `tp_ep_vram` and related metadata from profile artifacts and
writes JSON plus Markdown summaries.

## Validation

```text
python3 -m py_compile tools/ds4-v100-tp-ep-vram-ledger.py
git diff --check
```

Both passed.

Generated artifacts:

- `logs/from-cluster/sprint404-vram-ledger/vram-ledger.json`
- `logs/from-cluster/sprint404-vram-ledger/vram-ledger.md`

## Key Result

At the `1536 MiB` NCCL reserve, the Sprint 403 HC-current NCCL target is short:

| GPU | Free at `nccl_after_output_head` | Deficit |
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

| Checkpoint | Delta |
|---|---:|
| NCCL rank-buffer/communicator overhead | `+848` to `+944 MiB/GPU` |
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

## Decision

The next NCCL implementation sprint should pair:

1. lazy/on-demand output-head residency, and
2. streaming or shrinking GPU0 HC-control residency.

Output-head deferral alone is insufficient for GPU0. HC-control reduction
alone is insufficient for GPUs 1, 4, 5, and 6. The paired change is enough on
paper to move HC-current NCCL above the `1536 MiB` reserve at `32` slots /
`256K`.
