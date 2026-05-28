# Sprint 522 - Make TP/EP Profile Launch Surface Generic

Date: 2026-05-28

## Goal

Keep `tools/ds4-v100-tp-ep-profile.py` as a generic profile collector instead
of an experiment-specific flag mapper.

## Changes

- Removed stale promoted-gate profile flags from the parser.
- Removed no-op `DS4_V100_TP_EP_*` experiment env exports from `build_env`.
- Kept generic run labeling through `--run-description`.
- Kept explicit binary customization through repeated `--server-arg`.
- Reduced case-directory suffixes to run description, raw server-arg digest,
  CUDA device order, NCCL policy, and profiler device selection.

## Validation

- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-http-readiness-check.py tools/ds4-v100-http-response-parity.py tools/ds4-v100-http-response-tolerance.py tools/ds4-v100-tp-ep-reference-parity.py`
- `tools/ds4-v100-tp-ep-profile.py --help`
- Help grep confirmed removed experiment flags are absent.
- `git diff --check -- tools/ds4-v100-tp-ep-profile.py`

## Notes

- Specific appliance binary options should be passed as `--server-arg` tokens
  when profiling a new experiment.
