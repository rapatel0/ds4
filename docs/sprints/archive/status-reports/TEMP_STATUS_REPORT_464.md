# TEMP Status Report 464

## Topline

Parallel expert loading is now promoted as the TP/EP appliance startup default.

The validated Sprint 463 gate:

```text
DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=1
```

now defaults on in `tools/ds4-v100-run-appliance.sh`. Operators can still force
the old serial behavior with:

```text
DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=0
```

The profile and HTTP A/B wrappers also default to parallel expert loading now.
Use this only for a serial-load diagnostic:

```text
--disable-parallel-expert-load
```

## Why

The old path loaded expert residency in serial GPU round-robin chunks. The
parallel loader keeps layers sequential but loads all eight GPU expert bindings
concurrently per layer. The clean target run passed:

```text
32 requests / 32 slots / 256K context / 32 generated tokens
```

with:

| Metric | Result |
|---|---:|
| HTTP responses | 32/32 |
| readiness elapsed | 106.215634 s |
| server generated decode tok/s | 35.813083 |
| request-window avg GPU util | 12.534426% |
| min free VRAM | 1734 MiB |
| VRAM failures | 0 |

## Status

Promoted for startup/iteration speed only. This does not change the current
steady-state bottleneck:

| Domain | Share |
|---|---:|
| EP / routed FFN | 52.10% |
| HC-current input/staging | 43.17% |

Next performance work should continue on HC-current staging and routed FFN/EP
runtime cost.
