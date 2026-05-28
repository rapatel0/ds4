# Sprint 426: Rank-Major Router Logits

## Objective

Continue the TP/EP-only rank-major conversion by moving router-logit
generation away from device-0 full-hidden tensors.

No PP/layer-split work is in scope.

## Implementation

Added a default-off diagnostic gate:

```text
--model-router-rank-major-logits-gate
DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
```

When enabled:

- each rank keeps only its `32` local expert router columns;
- each rank computes router logits for its expert shard from rank-major hidden;
- ranks NCCL-allgather the small `slots x 256` logits tensor;
- device 0 reorders logits for the existing top-k selector;
- the full replicated device-0 router matrix is skipped under this gate.

The same helper is used for HC-current routing and post-attention routing. This
keeps the path rank-major instead of special-casing only post-attention FFN
input.

## Validation

V100 sm_70 build passed.

Resident layer 2:

```text
/localpool/ds4/workspace/logs/sprint426-rankmajor-router-logits/resident-layer2-router-rankmajor-final/
```

| Metric | Value |
|---|---:|
| rc | 0 |
| checksum | 4161861552 |
| graph capture/replay | pass |
| replay ms | 13.329408 |
| decode ms/step | 3.332352 |
| slot-step tok/s | 2400.706824 |

Prior same-binary resident references:

| Mode | Decode ms/step | Slot-step tok/s | Checksum |
|---|---:|---:|---:|
| post-FFN control | 3.391488 | 2358.846566 | 4161861552 |
| rank-major routed | 3.554816 | 2250.468093 | 4161861552 |
| rank-major router logits | 3.332352 | 2400.706824 | 4161861552 |

## Full-Layer Status

The `8` slot / `256K` all-layer run with dense F16 cache still OOMs during
expert residency:

```text
/localpool/ds4/workspace/logs/sprint424-rankmajor-router-logits/full-router-rankmajor-slot8-tokens4-scratch256-nofull-routerw/
```

Failure:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9182: out of memory
```

The earlier version also OOMed before skipping the full router matrix. Skipping
the full router matrix moved the run farther but did not make the full
all-layer cached-expert shape fit.

## Decision

Keep `--model-router-rank-major-logits-gate` default-off.

The kernel path is resident-correct and aligns with the rank-major strategy, but
it is not promotable until the full all-layer expert-residency memory problem is
resolved. This is now a memory-layout issue, not a reason to return to PP or
device-0 gather/compute/redistribute routing.

## Next

1. Reduce full all-layer expert residency so `8` slot / `256K` semantic
   post-attention runs can fit with rank-major router logits.
2. Re-run the Sprint 425 shared-only / routed-only parity split under the
   all-layer persistent graph harness.
3. Promote only after all-layer checksum or prompt-level parity is stable.
