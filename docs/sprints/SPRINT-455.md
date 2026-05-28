# Sprint 455: Admit 32-Slot Longer Rank-Major Serving With Scratch 1280

## Objective

Keep the same `32` requests / `32` slots / `256K` context / `32` generated
token window from Sprint 454, reduce TP runtime scratch from `1536 MiB` to
`1280 MiB`, and verify that the promoted router+FFN rank-major bundle still
improves throughput while passing readiness and response parity.

## Result

Artifact:

```text
/localpool/ds4/workspace/logs/s455-router-ffn-rankmajor-s32-t32-scratch1280
```

| Metric | Control | Candidate | Speedup |
|---|---:|---:|---:|
| Readiness | pass | pass | pass |
| Response parity | `32/32` | `32/32` | pass |
| Server generated decode tok/s | `33.170805` | `35.578211` | `1.0726x` |
| Server continuation tok/s | `33.156600` | `35.585793` | `1.0733x` |
| Client generated tok/s | `13.525258` | `14.801409` | `1.0944x` |
| Average GPU util | `10.24%` | `11.77%` | `1.1488x` |
| HC-current gather ms | `6.820885` | `6.191652` | `0.9077x` |
| HC-current input ms | `459.277972` | `399.228564` | `0.8693x` |
| Minimum free VRAM | `1584 MiB` | `1734 MiB` | `1.0947x` |
| VRAM failures vs 1536 MiB reserve | `0` | `0` | pass |

## Decision

Promote the target serving defaults:

```text
DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB=1280
DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1
DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
```

This is now the cleanest TP/EP serving baseline at `32` slots / `256K` context.
The remaining utilization ceiling is still low, so the next performance work
should target launch/sync/staging rather than more small FFN input rewrites.
