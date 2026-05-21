# Sprint 150 - Two-GPU TP Split Probe

Date: 2026-05-21

## Objective

Move from the one-GPU TP split estimate to a real two-GPU routed-FFN proxy on
the V100 node, using NV2-connected GPU pairs.

## Changes

- Added `test_ggml_turbomind_tp_split_2gpu`.
- The benchmark:
  - builds the same DS4 MXFP4 compact routed FFN fixtures as the gate/up
    harness;
  - runs the full gated gate/up plus down path on one GPU;
  - splits the `2048` intermediate dimension into two `1024` halves;
  - packs half 0 on one GPU and half 1 on a peer GPU;
  - launches both half-FFNs concurrently;
  - measures a conservative total that includes input activation copy to the
    peer and partial-output copy back.

## Results

Clean NV2 pairs:

| Pair | Routes | Full one-GPU FFN | Concurrent half compute | Compute speedup | Total with copies | Total speedup |
|---|---:|---:|---:|---:|---:|---:|
| `0,3` | 768 | `1.1623 ms` | `0.6221 ms` | `1.868x` | `0.9063 ms` | `1.282x` |
| `4,7` | 768 | `1.1606 ms` | `0.6207 ms` | `1.870x` | `0.9061 ms` | `1.281x` |
| `0,3` | 1536 | `1.3466 ms` | `1.0409 ms` | `1.294x` | `1.5814 ms` | `0.852x` |
| `4,7` | 1536 | `1.3858 ms` | `0.9488 ms` | `1.461x` | `1.4682 ms` | `0.944x` |

Payloads:

- 768 routes: `6 MiB` input copy plus `6 MiB` output partial copy.
- 1536 routes: `12 MiB` input copy plus `12 MiB` output partial copy.

## Interpretation

Two-way TP is real for the 768-route/128-slot shape if we use clean NV2 pairs:
the conservative proxy still shows about `1.28x` after copies. At the
1536-route/256-slot shape, the larger payload plus half-kernel imbalance makes
the conservative proxy neutral to slower.

This means TP should not be applied blindly across the current 8-GPU
layer-owned runtime. It is worth prototyping only where the route shape and
topology give a clear win, and it needs better overlap or replicated hidden
state before it can help the current 256-slot ceiling.

## Decision

Keep the production scheduler unchanged. The next TP step, if pursued, should
be an opt-in single-stage prototype on NV2 pairs for the 128-slot/32K tier,
with explicit correctness comparison against the current layer-owned FFN
output.

## Artifacts

- `logs/from-cluster/sprint150-tp-split-2gpu/`
