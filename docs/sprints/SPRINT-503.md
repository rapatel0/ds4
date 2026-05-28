# Sprint 503 - Trim TP/EP Appliance CLI Surface

Date: 2026-05-28

## Goal

Continue Phase 3 flag cleanup by removing promoted/default path switches from
tool invocations and from the public appliance help surface.

## Changes

- Removed promoted default flags from the sustained bench direct appliance
  invocation.
- Removed promoted default flags from the direct profile command base.
- Replaced the long appliance usage banner with runtime-facing knobs only.
- Kept explicit diagnostic/profile gates available for targeted investigation.

## Validation

- `git diff --check`
- `bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-run-tp-ep-appliance.sh tools/ds4-v100-tp-ep-http-bench.sh tools/ds4-v100-tp-ep-sustained-bench.sh`
- `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-steady-profile.py`
- Remote V100 build:
  `make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 appliance/ds4-v100-tp-ep-appliance`
- Remote startup:
  `./appliance/ds4-v100-tp-ep-appliance --help`
- Remote help audit confirms the promoted-path flags are not advertised:
  `--true-ds4`, `--tp-hc`, `--model-router`, `--fuse-compose-sum`,
  `--dense-f16`, `--all-layers`, `--shared-`, `--token-major`,
  `--multi-copy`, `--skip-descriptor`, `--skip-predecode`.
