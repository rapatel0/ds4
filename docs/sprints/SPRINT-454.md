# Sprint 454: Longer Router+FFN Rank-Major Serving Window

## Objective

Recheck the promoted router+FFN rank-major TP/EP bundle with the target
`32` slots / `256K` context shape over a longer `32` token serving window.

## Result

Artifact:

```text
/localpool/ds4/workspace/logs/s454-router-ffn-rankmajor-s32-t32
```

The run preserved response parity but failed the strict readiness gate because
both legs dipped below the `1536 MiB` minimum-free VRAM reserve.

| Metric | Control | Candidate | Speedup |
|---|---:|---:|---:|
| Response parity | `32/32` | `32/32` | pass |
| Server generated decode tok/s | `33.341678` | `35.303611` | `1.0588x` |
| Server continuation tok/s | `33.331332` | `35.305162` | `1.0592x` |
| Client generated tok/s | `14.248305` | `14.796928` | `1.0385x` |
| Average GPU util | `11.04%` | `11.76%` | `1.0650x` |
| Minimum free VRAM | `1328 MiB` | `1478 MiB` | `1.1130x` |
| VRAM failures vs 1536 MiB reserve | `62` | `32` | better, not clean |

## Decision

Do not use this as the final promotion record because readiness failed. The
throughput signal is real and parity-clean, but the `1536 MiB` scratch default
is too large for the longer target run with the strict reserve. Continue with a
scratch-size admission sprint.
