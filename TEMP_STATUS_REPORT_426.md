# TEMP Status Report 426

Date: 2026-05-27

## Focus

Rank-major router logits for TP/EP. The goal is to remove another
`gather full hidden to device 0 -> compute -> redistribute` pattern.

## What Changed

Implemented:

```text
--model-router-rank-major-logits-gate
DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
```

The new path:

- stores router weights as per-rank expert shards;
- computes local expert logits from rank-major hidden on every GPU;
- allgathers only the small logits tensor;
- skips the replicated device-0 full router matrix when this gate is enabled;
- works for both HC-current and post-attention router points.

Launcher, env example, profiler CLI, and scaffold summary output are wired.

## Current Evidence

Build:

```text
V100 sm_70 build passed
```

Resident layer 2 final artifact:

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

Comparison against current resident references:

| Mode | Decode ms/step | Slot-step tok/s | Checksum |
|---|---:|---:|---:|
| post-FFN control | 3.391488 | 2358.846566 | 4161861552 |
| rank-major routed | 3.554816 | 2250.468093 | 4161861552 |
| rank-major router logits | 3.332352 | 2400.706824 | 4161861552 |

## Full All-Layer Result

The full `8` slot / `256K` semantic post-attention run still OOMs at expert
residency:

```text
/localpool/ds4/workspace/logs/sprint424-rankmajor-router-logits/full-router-rankmajor-slot8-tokens4-scratch256-nofull-routerw/
```

Error:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9182: out of memory
```

This is after removing the full replicated device-0 router matrix under the new
gate. The remaining blocker is full all-layer expert/dense residency headroom.

## Decision

Default remains off.

This is the right topology direction and it is resident-correct, but not
production-promotable until the all-layer memory fit and Sprint 425 parity split
are resolved.

## Next

1. Fix all-layer expert residency/headroom for semantic post-attention rank-major
   runs.
2. Re-run shared-only and routed-only rank-major FFN input probes.
3. Re-run all-layer checksum/prompt parity.
4. Only then run HTTP promotion.
