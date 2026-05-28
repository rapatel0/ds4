# TEMP Status Report 482 - Code Cleanup Deeper Split

Date: 2026-05-28

## Scope

Continued the code-cleanup sprint from `TEMP_CODE_CLEANUP_PROMPT.md`.
This pass went one level deeper on two cleanup axes:

- Remove rejected/default-off TP/EP gated code from active serving paths.
- Introduce mode-specific PP and TP/EP launcher entrypoints so active tooling no
  longer relies on `DS4_V100_SERVE_MODE` selection.

## Changes

- Removed the rejected `--async-output-gate` surface:
  - binary option and `Options` field
  - async output-head execution branch
  - launcher env/config/command plumbing
  - profile and active-slot matrix flags/suffixes
  - deployment example env
- Removed the rejected `--batched-paged-attn-gate` surface:
  - binary option and `Options` field
  - diagnostic row-plan structs/helpers/logging
  - launcher env/config/command plumbing
  - profile and active-slot matrix flags/suffixes
- Added separate launcher entrypoints:
  - `tools/ds4-v100-run-tp-ep-appliance.sh`
  - `tools/ds4-v100-run-pp-appliance.sh`
- Updated TP/EP harnesses to use the TP/EP launcher:
  - `tools/ds4-v100-tp-ep-profile.py`
  - `tools/ds4-v100-tp-ep-http-ab.py`
  - `tools/ds4-v100-tp-ep-http-bench.sh`
- Updated PP/production baseline scripts to use the PP launcher.
- Updated VISION cleanup notes for the removed rejected gates.

## Checks

- `bash -n` passed for modified shell launchers and gate scripts.
- `python3 -m py_compile` passed for modified Python harnesses.
- `git diff --check -- tools deploy docs/sprints/VISION.md` passed.
- Local `--check --allow-missing` smoke passed for both mode-specific
  launchers and confirmed `mode=base` for PP and `mode=tp-ep` for TP/EP.
- Local direct Makefile CUDA target is the non-CUDA stub and exits with
  `tools/ds4-v100-tp-ep-full-layer-smoke requires a CUDA build`.
- Pod CUDA build passed in `llm/ds4-tp-bench` after copying the modified CUDA
  file to `/workspace/ds4-sprint181`:

```bash
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

The build emitted only the pre-existing unused-kernel warnings for
`rms_norm_plain_rows_kernel` and `indexer_score_row0_slots_kernel`.

## Notes

- No serving revalidation was run in this pass, per instruction to keep this to
  build checks until final validation.
- `research/` remains untouched and ignored for this cleanup pass.
