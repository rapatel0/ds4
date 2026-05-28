# Sprint 501 - Retire TP/EP Full-Layer Smoke

Date: 2026-05-28

## Goal

Complete the smoke-retirement portion of the structural extraction plan for the
TP/EP serving path.

## Changes

- Deleted `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Removed the full-layer smoke Makefile targets.
- Updated deploy defaults and TP/EP bench/profile tools to use
  `appliance/ds4-v100-tp-ep-appliance`.
- Kept historical sprint documentation unchanged as run lineage.

## MTP Scope

MTP remains separate unfinished work and is not promoted by this cleanup.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-run-tp-ep-appliance.sh tools/ds4-v100-tp-ep-http-bench.sh tools/ds4-v100-tp-ep-sustained-bench.sh`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-steady-profile.py`
- Launcher print-command check resolves to
  `./appliance/ds4-v100-tp-ep-appliance`.
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Remote smoke retirement check:
  `test ! -e tools/ds4-v100-tp-ep-full-layer-smoke.cu && test ! -e tools/ds4-v100-tp-ep-full-layer-smoke`
