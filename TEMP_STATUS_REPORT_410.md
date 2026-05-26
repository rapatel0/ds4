# TEMP Status Report 410

Current focus: TP/EP NCCL serving path only. No PP/layer-split work.

## Result

Sprint 410 completed a target-shape HTTP A/B for the HC-current NCCL
allgather path.

Artifact:

```text
logs/from-cluster/sprint410-nccl-http-ab/
```

Shape:

```text
32 concurrent HTTP requests
32 configured slots
262144 context
32 generated tokens/request
position 262080
lazy output head
compact MoE decode
model-router routes
skip unused TP-runtime comp-state
```

## Topline

| Metric | Control | HC-current NCCL |
|---|---:|---:|
| HTTP 200 | 32/32 | 32/32 |
| response parity | 32/32 | 32/32 |
| first token | 83484 | 83484 |
| server generated decode tok/s | 101.897890 | 107.723452 |
| server continuation decode tok/s | 101.682616 | 107.545644 |
| client generated tok/s | 17.223947 | 16.627120 |
| avg sampled GPU util | 4.535714% | 3.524272% |
| max sampled GPU util | 49% | 47% |
| min free VRAM | 2738 MiB | 2106 MiB |
| VRAM failures | 0 | 0 |

Decision: promote HC-current NCCL as the TP/EP appliance default because it is
correct, memory-admitted at the target shape, and improves server decode by
about `5.7%`.

## Caveat

Client generated throughput regressed by about `3.5%` and sampled average GPU
utilization is still very low. This means the narrow HC-current NCCL boundary
is not the main utilization lever. The next optimization should target broader
TP/EP collective placement and request/decode orchestration, not more PP work.

## Code Changes

- Added `tools/ds4-v100-tp-ep-nccl-http-ab.py`.
- Promoted `DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1` in the appliance
  launcher default and env example.
- Kept profile harness flags explicit so future A/Bs can still force control
  and candidate variants.
