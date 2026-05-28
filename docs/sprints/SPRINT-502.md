# Sprint 502 - Promote TP/EP Appliance Defaults

Date: 2026-05-28

## Goal

Continue Phase 3 flag cleanup by moving launcher-supplied promoted TP/EP path
switches into appliance defaults. The production launcher should no longer
construct serving by passing a long matrix of `--*-gate` flags.

## Changes

- Promoted TP/EP serving switches now default on in `appliance/options.cu`.
- `tools/ds4-v100-run-appliance.sh` now passes only appliance paths, shape,
  HTTP endpoint, tokenizer, VRAM/NCCL admission, profiler, CUDA graph debug
  options, and explicit extra args.
- MTP remains unsupported for TP/EP serving and unchanged.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-run-tp-ep-appliance.sh tools/ds4-v100-tp-ep-http-bench.sh tools/ds4-v100-tp-ep-sustained-bench.sh`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-steady-profile.py`
- Launcher print-command check now emits no promoted-path gate arguments in
  the default TP/EP command.
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
