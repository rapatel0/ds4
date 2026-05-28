# TEMP Status Report 463

## Topline

The observed startup pattern was real: expert residency was being loaded in a
serial GPU round-robin pattern, with roughly 500 MB-scale chunks appearing on
one GPU at a time. I added an opt-in parallel expert-load path that loads the
gated and down expert packs for all 8 GPUs concurrently per layer.

This is a startup and iteration-speed fix, not the decode throughput fix.

## Implementation

New opt-in gate:

```text
--parallel-expert-load-gate
DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=1
```

Touched components:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
tools/ds4-v100-run-appliance.sh
tools/ds4-v100-tp-ep-profile.py
tools/ds4-v100-tp-ep-nccl-http-ab.py
```

The implementation keeps layer order sequential, but fans out the per-GPU
expert pack loads inside each layer. That avoids the old GPU0-to-GPU7
round-robin residency fill while keeping peak VRAM predictable.

## Validation

Clean 8-slot smoke:

```text
/localpool/ds4/workspace/logs/s460-parallel-expert-load-s8-t3-clean
```

| Metric | Result |
|---|---:|
| HTTP responses | 8/8 |
| readiness elapsed | 53.110685 s |
| request elapsed | 11.508914 s |
| server generated decode tok/s | 20.688739 |
| server continuation decode tok/s | 20.695143 |
| request-window avg GPU util | 11.867925% |
| min free VRAM | 5092 MiB |
| VRAM failures | 0 |

32-slot target-window run:

```text
/localpool/ds4/workspace/logs/s460-parallel-expert-load-s32-t32-long
```

Shape:

```text
32 requests / 32 slots / 256K context / 32 generated tokens
```

| Metric | Result |
|---|---:|
| HTTP responses | 32/32 |
| generated tokens | 1024 |
| readiness elapsed | 106.215634 s |
| request elapsed | 67.038542 s |
| server generated decode tok/s | 35.813083 |
| server continuation decode tok/s | 35.815690 |
| client generated tok/s | 15.274795 |
| full-run avg GPU util | 5.246531% |
| startup avg GPU util | 1.729430% |
| request-window avg GPU util | 12.534426% |
| request-window max GPU util | 32.000000% |
| moving-average peak GPU util | 16.000000% |
| peak phase | request |
| min free VRAM | 1734 MiB |
| max used VRAM | 30759 MiB |
| VRAM failures | 0 |

Expert residency load line:

```text
tp_ep_shared_expert_bindings_load layers 43 parallel 1 bytes 147169738752 load_ms 81460.580937 PASS
```

## Current Bottleneck

The 32-slot decode bottleneck did not move. The request-window timing is still
dominated by the same two domains:

| Domain | ms | Share |
|---|---:|---:|
| EP / routed FFN | 473.477984 | 52.10% |
| HC-current input/staging | 392.333944 | 43.17% |
| final HC | 23.094316 | 2.54% |
| compose | 19.914499 | 2.19% |

Top fine-grained contributors:

| Fine domain | ms | Share |
|---|---:|---:|
| pre-EP HC-current | 91.347854 | 10.05% |
| pre-EP compressed KV | 75.136642 | 8.27% |
| pre-EP post-attention FFN input | 62.972578 | 6.93% |
| pre-EP attention projection | 53.635911 | 5.90% |
| HC-current FFN router | 48.538947 | 5.34% |
| pre-EP attention state | 41.366386 | 4.55% |
| pre-EP attention output | 39.133127 | 4.31% |
| HC-current route upload | 35.220289 | 3.88% |

## Interpretation

The parallel load gate fixes the visible sequential startup behavior and makes
future benchmarks faster to iterate. It should remain opt-in until we run one
more same-binary startup A/B or decide startup speed is enough to enable by
default.

Serving performance is still limited by runtime EP work plus HC-current staging.
The next performance work should stay focused on reducing rank-major boundary
movement and routed FFN/EP cost, not on layer-parallel variants.
