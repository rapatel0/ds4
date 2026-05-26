# TEMP_STATUS_REPORT_392

Date: 2026-05-25

## Focus

Sprint 392 added multi-prompt soak support to the TP/EP HTTP profiler and used
it to re-test E5M2 KV beyond the repeated prompt template.

## Changes

- `tools/ds4-v100-tp-ep-profile.py` now accepts `--prompt-file`.
- Prompt files are JSONL records with either `prompt` or `messages`.
- Profile summaries now include `prompt_file`, `prompt_count`, and
  `prompt_digest`.
- Added `tests/v100_tp_ep_soak_prompts.jsonl` with `16` varied prompts.

## V100 Result

Shape:

```text
32 requests
32 slots
32 generated tokens/request
256K context
position=262080
model-router routes
compact MoE
VRAM admission enabled
```

| Metric | Control | E5M2 KV |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| Prompt count | `16` | `16` |
| First token | `83484` | `83484` |
| Parity matched pairs | `32/32` | `32/32` |
| Client generated tok/s | `38.912861` | `39.774181` |
| Server generated tok/s | `88.358577` | `88.351220` |
| Server decode tok/s | `106.390802` | `106.483285` |
| Compressed-KV sum | `3343.550356 ms` | `3301.691102 ms` |
| Avg GPU util | `9.772727%` | `9.691860%` |
| Max GPU util | `52%` | `50%` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

Permanent parity comparator:

```text
match=true
paired_count=32
matched_pairs=32
failed_pairs=0
```

## Decision

E5M2 remains diagnostic-only. The multi-prompt run is parity-clean, but
throughput is effectively flat and the current E5M2 row layout is not a memory
capacity improvement over E4M3. Do not take the precision tradeoff as a
production default yet.

## Artifacts

- `logs/from-cluster/sprint392-multiprompt-e5m2`
  - `http-control`
  - `http-candidate-e5m2`
  - `http-parity-summary.json`
