# Sprint 520 - Remove Stale TP/EP HTTP A/B Harness

Date: 2026-05-28

## Goal

Delete the obsolete typed-KV TP/EP HTTP A/B harness now that the appliance no
longer consumes its experiment controls or emits the telemetry fields it reads.

## Changes

- Removed `tools/ds4-v100-tp-ep-http-ab.py`.
- Left shared lock-name strings in the active TP/EP correctness, NCCL, and
  steady-profile helpers unchanged.

## Validation

- `rg -n "ds4-v100-tp-ep-http-ab\\.py|tp-ep-http-ab" Makefile tools deploy appliance engine smokes docs/sprints/SPRINT-5*.md docs/sprints/STATUS.md`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py tools/ds4-v100-tp-ep-correctness-gate.py tools/ds4-v100-tp-ep-steady-profile.py`
- `git diff --check -- tools/ds4-v100-tp-ep-http-ab.py`

## Notes

- The only remaining filename references are historical sprint notes.
- The remaining `tp-ep-http-ab` matches are lock-file names used by active
  helpers, not calls into the deleted harness.
