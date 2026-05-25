# TEMP Status Report 377: Batched Paged Attention Gate

Date: 2026-05-25

## Current Focus

Sprint 377 is the S-C gate from `TEMP_THROUGHPUT_PROMPT.md`:
`--batched-paged-attn-gate`.

Sprint 376 rejected CUDA graph replay because the current TP/EP decode step
uses stream-capture-incompatible `cudaMemcpyPeerAsync` transport. Sprint 377
therefore targets launch-count and staging reduction in the attention/KV row
path without depending on CUDA graphs.

## Baseline V100 Run

Command shape:

```text
32 active chat requests
32 configured slots
256K context
position 262080
32 generated tokens/request
GPU sampling interval 250 ms
```

Artifact path:

```text
logs/from-cluster/sprint377-batched-paged-attn/baseline-matrix
```

Topline:

| Metric | Value |
|---|---:|
| HTTP 200 | `32/32` |
| Coalesced batch size | `32` |
| First token | `89340` |
| Client generated tok/s | `40.157540` |
| Server generated tok/s | `74.895420` |
| Server generated tok/s decode | `88.372350` |
| Server continuation tok/s decode | `88.329223` |
| Scaffold projected slot-step tok/s | `56.990488` |
| Avg GPU util | `7.972222%` |
| Max GPU util | `38%` |
| Max GPU memory used | `32398 MiB` |
| Compressed-KV sum | `5436.764269 ms` |

Stage evidence from the first scaffold row:

| Stage | ms |
|---|---:|
| `sum_decode_ms` | `561.497209` |
| `sum_pre_ep_attention_projection_ms` | `87.215132` |
| `sum_pre_ep_attention_state_ms` | `103.187385` |
| `sum_pre_ep_compressed_kv_ms` | `162.085447` |
| `sum_pre_ep_raw_read_ms` | `40.072128` |
| `sum_pre_ep_typed_history_ms` | `15.094877` |
| `sum_hc_current_input_ms` | `457.899720` |
| `sum_ep_ms` | `31.515356` |
| `sum_compose_ms` | `42.200063` |

Interpretation: baseline utilization is still low and GPU0-heavy. The
attention/KV prefix remains a large part of the decode step, with compressed
KV and attention state/projection the most visible targets for this sprint.

## Next Work

1. Add default-off `--batched-paged-attn-gate` plumbing.
2. Add launcher/profile env plumbing.
3. Implement a fixed-size per-layer row-family plan for raw-SWA, compressed
   attention, and ratio-4 indexer rows.
4. Add the smallest batched row-family kernel that can reduce launch count
   while preserving first-token/checksum parity.
5. Build and run direct + HTTP V100 A/B.
