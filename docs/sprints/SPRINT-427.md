# Sprint 427: Rank-Major FFN Half-Input Parity Audit

## Objective

Replace end-to-end checksum guessing for the rank-major post-attention FFN
input path with direct pre-consumer half-input parity probes.

No PP/layer-split work is in scope.

## Context

Sprint 425 split `--routed-ffn-rank-major-input-gate` into shared-only and
route-only gates:

- shared-only first diverges at step 0 layer 0
- route-only matches layer 0 and first diverges at step 0 layer 1
- combined follows the shared-only layer-0 failure

That localizes the blocker, but still does not say whether the bad value is
caused by rank-major normalization, rank-major indexing, half conversion, route
slot metadata, or a later consumer.

## Implementation

Add a default-off diagnostic gate:

```text
--routed-ffn-rank-major-input-parity-gate
```

When enabled under the post-attention FFN input path:

- keep the legacy slot-major `hc->d_ffn_normed -> r.d_current_full` copy
  available for comparison;
- after rank-major shared gate/up input fill, compare the produced half buffers
  against the legacy slot-major half conversion of `r.d_current_full`;
- after rank-major route packing, compare `r.d_a` against the legacy route pack
  implied by `r.d_current_full` and `r.d_route_slots`;
- emit one compact line per layer/rank family with mismatch count and max
  absolute difference in half-as-float space;
- keep the gate diagnostic-only and default-off.

## Definition of Done

- V100 sm_70 build passes.
- The parity audit gate is visible in CLI usage and scaffold logging.
- An `8` slot / `256K` all-layer run with shared-only + parity audit records
  direct shared input diffs.
- An `8` slot / `256K` all-layer run with route-only + parity audit records
  direct routed input diffs.
- The sprint records the first layer/rank/family with mismatched half inputs.
- No rank-major FFN input gate is promoted unless direct half-input parity is
  clean and end-to-end checksum improves or matches.

## Outcome

V100 sm_70 build passed.

The new gate is available as:

```text
--routed-ffn-rank-major-input-parity-gate
```

It logs:

```text
tp_ep_rank_major_half_input_diff
```

The diagnostic could not run with `--route-plan-async-upload-gate` in eager
mode because that path intentionally rejects `routed_ffn_norm_input_gate`
outside the graph/reuse regime. I therefore ran the parity audit with the
legacy synchronous route-plan upload and CUDA graph replay disabled.

Final artifacts:

```text
/localpool/ds4/workspace/logs/sprint427-rankmajor-half-input-parity-syncplan/
```

Shape:

```text
slots=8
ctx=262144
decode_steps=1
tp_runtime_scratch_mib=128
HC-current NCCL allgather=on
post-attention FFN input=on
semantic stats skip=on
CUDA graph replay=off
route-plan async upload=off
```

## Results

Direct half-input parity:

| Case | Diff lines | Mismatch lines | Families | Result |
|---|---:|---:|---|---|
| shared-only | 688 | 0 | `shared_gate`, `shared_up` | clean |
| route-only | 329 | 0 | `route_a` | clean |

Family summaries:

| Case | Family | Lines | Mismatches | Max abs |
|---|---|---:|---:|---:|
| shared-only | `shared_gate` | 344 | 0 | 0 |
| shared-only | `shared_up` | 344 | 0 | 0 |
| route-only | `route_a` | 329 | 0 | 0 |

Same-mode end-to-end checksums:

| Case | Decode tok/s | Decode ms | Checksum |
|---|---:|---:|---:|
| control | 12.525670 | 638.688378 | 8358757728 |
| shared-only + parity | 10.650018 | 751.172463 | 8358757728 |
| route-only + parity | 12.356142 | 647.451259 | 8358757728 |

This is the key result: in the synchronous-plan eager diagnostic regime, both
rank-major half-input variants are byte-identical to the legacy half inputs and
the all-layer checksum matches control.

## Decision

Do not promote the rank-major FFN input gates yet, because the previous
persistent-graph / async-route-plan regime still diverges.

But stop treating the shared/route half-input kernels as the primary blocker.
They are clean under direct pre-consumer comparison.

The next sprint should isolate the graph/async-route-plan interaction:

- determine whether route metadata differs between synchronous upload and async
  upload under graph capture/replay;
- add graph-safe device-resident route/half-input audit counters if needed;
- rerun the rank-major shared/route split in the exact persistent-graph regime
  that previously diverged.
