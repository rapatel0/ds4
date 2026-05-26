# TEMP_STATUS_REPORT_389

Date: 2026-05-25

## Focus

Sprint 389 revalidated the existing TP/EP
`--true-ds4-compressed-kv-skip-dense-stats-gate` against the current
real-router compact-MoE baseline at the target `32` slot / `256K` shape.

## Result

Promoted.

`tools/ds4-v100-run-appliance.sh` now defaults:

```text
DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1
```

The path remains disableable with:

```text
DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=0
```

The permanent profile harness is aligned with that production default:

```text
tools/ds4-v100-tp-ep-profile.py --disable-skip-compressed-dense-stats
```

now provides the control path for future same-binary A/B runs.

Launcher proof on gpu-01:

```text
unset/default: --true-ds4-compressed-kv-skip-dense-stats-gate present
explicit 0:   gate absent
```

## Direct A/B

Shape:

```text
32 slots
256K context
position=262080
32 decode steps
model-router routes
compact MoE
VRAM admission enabled with 64 MiB reserve
```

| Metric | Control | Skip stats |
|---|---:|---:|
| First token | `98751` | `98751` |
| Generated decode tok/s | `91.869507` | `102.871437` |
| Generated wall tok/s | `62.314303` | `70.068457` |
| Compressed-KV sum | `3138.980697 ms` | `1798.907552 ms` |
| Pre-EP compressed-KV | `3151.868435 ms` | `1812.357566 ms` |
| Total decode | `11146.244652 ms` | `9954.172169 ms` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

## HTTP Chat A/B

Shape:

```text
32 concurrent chat requests
32 configured slots
256K context
position=262080
32 generated tokens/request
model-router routes
compact MoE
GPU utilization sampling enabled
VRAM admission enabled with 64 MiB reserve
```

| Metric | Control | Skip stats |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| First token | `83484` | `83484` |
| Response checksum | `17913667583206000416` | `17913667583206000416` |
| Client generated tok/s | `42.183007` | `44.592824` |
| Server generated tok/s | `76.432156` | `86.367455` |
| Server decode tok/s | `89.709430` | `103.758804` |
| Compressed-KV sum | `5063.395601 ms` | `2835.901361 ms` |
| Pre-EP compressed-KV | `101.306501 ms` | `53.659646 ms` |
| Avg GPU util | `8.621875%` | `9.003289%` |
| Max GPU util | `43%` | `49%` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

## Parity

- Direct first token matched.
- HTTP first token matched.
- All `32` HTTP responses had identical generated token sequences.
- All `32` HTTP responses had checksum `17913667583206000416`.
- The response comparator found `4/32` raw semantic metadata diffs caused by
  request/cache metadata; generated text and token IDs matched.

## Artifacts

Cluster:

- `/workspace/logs/sprint389-skip-dense-stats/direct-control`
- `/workspace/logs/sprint389-skip-dense-stats/direct-candidate`
- `/workspace/logs/sprint389-skip-dense-stats/http-control`
- `/workspace/logs/sprint389-skip-dense-stats/http-candidate`

Local:

- `logs/from-cluster/sprint389-skip-dense-stats`

## Next

Continue TP/EP-only performance work. The next useful target is still the
launch/scheduling-heavy compressed/attention path and GPU0-heavy orchestration;
MTP remains deferred until base TP/EP serving metrology is stable enough that it
will expose a real multiplier instead of hiding bottlenecks.
