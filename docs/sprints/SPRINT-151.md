# Sprint 151 - Two-GPU TP Correctness Gate

Date: 2026-05-21

## Objective

Promote the two-GPU TP split proxy from timing-only evidence to a
correctness-bearing one-stage prototype.

## Changes

- Added full-output versus TP-split output comparison to
  `test_ggml_turbomind_tp_split_2gpu`.
- The benchmark now:
  - runs the full one-GPU gated gate/up plus down path;
  - runs the two `1024`-wide TP halves on an NV2 pair;
  - copies the peer partial back;
  - compares `full_down` against `half0_down + half1_down` in FP32;
  - fails the benchmark if the comparison exceeds the configured low-bit
    tolerance.
- Tightened the synthetic MXFP4 scale range in this correctness probe so the
  random fixture produces finite down outputs. This does not change the
  production pack path or appliance runtime.

## Results

Clean NV2 pairs with finite correctness fixtures:

| Pair | Routes | Total with copies | Total speedup | Correctness |
|---|---:|---:|---:|---|
| `0,3` | 768 | `0.9022 ms` | `1.281x` | PASS, `rel=2.4696e-04`, `bad=0` |
| `4,7` | 768 | `0.9013 ms` | `1.282x` | PASS, `rel=2.4696e-04`, `bad=0` |
| `0,3` | 1536 | `1.4803 ms` | `0.926x` | PASS, `rel=2.4622e-04`, `bad=0` |
| `4,7` | 1536 | `1.5254 ms` | `0.874x` | PASS, `rel=2.4622e-04`, `bad=0` |

Maximum absolute difference was `6.1035e-05` for all passing runs.

## Interpretation

The TP split math is correct for the one-stage routed-FFN proxy. The remaining
problem is scheduling and payload economics, not decomposition correctness.

The 768-route shape remains the only positive candidate with the conservative
copy model. The 1536-route shape is correct but slower after payload movement.

## Decision

The next production-facing TP step should target only the 128-slot/32K route
shape first. Implement it behind an explicit opt-in and compare the layer FFN
output against the current layer-owned path before wiring it into served
generation.

## Artifacts

- `logs/from-cluster/sprint151-tp-split-correctness/`
