# TEMP_STATUS_REPORT_391

Date: 2026-05-25

## Focus

Sprint 391 ran a longer E5M2 typed-KV parity/performance A/B at the real
`32` slot / `256K` TP/EP serving shape.

## Implementation Fix

Fixed `tools/ds4-v100-tp-ep-profile.py` direct-token-major command generation.
HTTP mode already inherited the promoted skip-dense-stats default, but direct
mode only emitted the gate when `--skip-compressed-dense-stats` was explicit.
Direct profiles now emit `--true-ds4-compressed-kv-skip-dense-stats-gate`
unless `--disable-skip-compressed-dense-stats` is set.

## Direct A/B

| Metric | Control | E5M2 KV |
|---|---:|---:|
| First token | `98751` | `98751` |
| Generated decode tok/s | `103.237368` | `102.152512` |
| Generated wall tok/s | `65.559769` | `69.125275` |
| Compressed-KV sum | `1808.998742 ms` | `1799.161576 ms` |
| Pre-EP compressed-KV | `1821.203311 ms` | `1812.406574 ms` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

## HTTP Chat A/B

| Metric | Control | E5M2 KV |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| First token | `83484` | `83484` |
| Parity matched pairs | `32/32` | `32/32` |
| Client generated tok/s | `46.115999` | `47.895831` |
| Server generated tok/s | `84.302469` | `89.226807` |
| Server decode tok/s | `101.206458` | `107.281060` |
| Compressed-KV sum | `2882.657866 ms` | `2678.431998 ms` |
| Pre-EP compressed-KV | `56.576454 ms` | `50.659014 ms` |
| Avg GPU util | `9.231618%` | `10.003571%` |
| Max GPU util | `49%` | `50%` |
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

E5M2 KV stays diagnostic-only. The HTTP result is positive and exact response
parity passed, but direct decode regressed slightly and E5M2 has a real
mantissa-precision risk. Promotion should require a broader multi-prompt
parity/soak run.

## Artifacts

- `logs/from-cluster/sprint391-e5m2-kv`
  - `direct-control-default`
  - `direct-candidate-e5m2`
  - `http-control-default`
  - `http-candidate-e5m2`
  - `http-parity-summary.json`
