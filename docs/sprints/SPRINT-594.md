# Sprint 594 - MTP raw-SWA activation capture

Date: 2026-05-30

## Why This Sprint Exists

Sprints 590-593 cleared the high-level MTP suspects: output-head HC slicing,
layer-43-only inverse attention-head RoPE, F8 dense pack/orientation, and
HC-current control/reduction. Acceptance remains `0/71`. The next boundary is
actual activation/state semantics inside the MTP body, especially raw-SWA,
attention output, post-attention handoff, and FFN output.

This sprint deliberately captures activation samples, not just aggregate stats.
The objective is to see whether any layer-43 activation boundary is obviously
wrong, frozen, stale, or insensitive, and to test the one raw-SWA addressing
hypothesis that can be checked cheaply.

## Scope

1. Add temporary MTP-only activation logging around `run_layer(43)`:
   - prologue output / layer-43 input HC shard.
   - current raw-SWA row contents after the layer writes it.
   - attention-head output after raw-SWA read/window.
   - post-attention shard.
   - next hidden and final HC shard after the body.
2. Log a small fixed sample of real activation values for rank 0 / slot 0
   alongside `step`, `position`, `mtp_raw_valid_rows`, and physical raw row.
3. Build and run the deterministic MTP acceptance harness on the pod.
4. Run one temporary raw-row addressing probe if the activation log suggests
   position/count mismatch: use the MTP raw-count frontier as the layer-43
   physical position while keeping serving output unchanged.
5. Remove temporary logging/probe code before commit unless a minimal durable
   correctness fix is found.

## Non-Goals

- Do not add a permanent diagnostic flag.
- Do not run throughput A/B.
- Do not build the K-wide verifier until acceptance is nonzero.
- Do not re-test dense F8 or HC-control probes unless their inputs change.

## Definition of Done

- Temporary activation logging build passes on the pod.
- Deterministic harness emits activation samples for MTP layer 43.
- The result records whether activations are finite, input-responsive, and
  whether the raw row/position frontier looks coherent.
- Any raw-row addressing probe has deterministic acceptance evidence.
- Temporary code is removed before commit unless promoted as a correctness fix.
- `VISION.md` and `SPIKE_B_STEERING.md` record the result and the next target.

## Result

Status: COMPLETE. No code was promoted.

Temporary MTP-only activation logging was added around the layer-43 draft body
and built on the pod. The deterministic acceptance harness still rejected the
draft path:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12]  [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]
```

The activation samples were finite and input-responsive across the tested
prompts. The logged raw-SWA frontier was coherent at the sampled boundaries:
positions `0`, `1`, `2`, `3`, `7`, `15`, and `31` mapped to physical raw rows
`0`, `1`, `2`, `3`, `7`, `15`, and `31`, with valid-row counts `1`, `2`, `3`,
`4`, `8`, `16`, and `32`.

Representative rank-0/slot-0 samples:

```text
pos0 valid1  prologue sum_abs=2.19080415  raw_swa sum_abs=12.4375  attn_heads sum_abs=0.136830302  body_final_hc sum_abs=12.8446698
pos31 valid32 prologue sum_abs=1.11315639 raw_swa sum_abs=12.125   attn_heads sum_abs=5.09488309   body_final_hc sum_abs=21.400773
```

This rules out the obvious stale/frozen/zero activation class and does not
support the cheap raw-row addressing probe. The current evidence points to a
subtler layer-43 semantic mismatch below the already-cleared dense-pack,
output-head, and HC-control paths: raw-SWA attention math/window semantics,
attention-output handoff, or routed-FFN activation sequence.

The temporary activation logging was removed from `engine/token_major_loop.cu`.
The clean source was recopied to `/workspace/s573-continuation-instrument`, and
the clean pod build passed:

```text
BUILD_EXIT=0
make appliance/ds4-v100-tp-ep-appliance
```
