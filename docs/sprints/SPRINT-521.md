# Sprint 521 - Prune Stale TP/EP Promotion Wrappers

Date: 2026-05-28

## Goal

Remove one-off TP/EP A/B and matrix orchestration scripts that encode old
experiment-gate profiles. Keep the active tooling generic and centered on the
launcher, HTTP bench, profile collector, readiness/parity checks, sustained
resident bench, and reference parity.

## Changes

- Removed obsolete promotion/matrix wrappers:
  - `tools/ds4-v100-tp-ep-active-slot-matrix.py`
  - `tools/ds4-v100-tp-ep-correctness-gate.py`
  - `tools/ds4-v100-tp-ep-nccl-http-ab.py`
  - `tools/ds4-v100-tp-ep-nccl-kv-matrix.py`
  - `tools/ds4-v100-tp-ep-steady-profile.py`
  - `tools/ds4-v100-tp-ep-true-attn-http-ab.py`
  - `tools/ds4-v100-tp-experts-ab.py`

## Validation

- `rg -n "ds4-v100-tp-ep-(active-slot-matrix|correctness-gate|steady-profile|true-attn-http-ab|nccl-http-ab|nccl-kv-matrix)|ds4-v100-tp-experts-ab" Makefile tools deploy appliance engine smokes docs/sprints/SPRINT-5*.md docs/sprints/STATUS.md`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-http-readiness-check.py tools/ds4-v100-http-response-parity.py tools/ds4-v100-http-response-tolerance.py tools/ds4-v100-tp-ep-reference-parity.py`
- `bash -n tools/ds4-v100-tp-ep-http-bench.sh tools/ds4-v100-tp-ep-sustained-bench.sh tools/ds4-v100-run-appliance.sh tools/ds4-v100-run-tp-ep-appliance.sh`

## Notes

- Remaining matches are historical sprint/status lineage only.
- This change does not remove log or sprint documentation lineage.
