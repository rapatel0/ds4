# Sprint 284 - TP/EP Compact Route Compose

Date: 2026-05-23

## Goal

Reduce staged FP32 contribution traffic in the TP/EP compose path without
changing model precision or falling back to the rejected FP16 return path.

## Implementation

Updated `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

- Added `--compact-route-compose`.
- Added route-major EP contribution packing:
  - old layout: `[dest][slot][hidden_shard]`
  - compact layout: `[dest][route][hidden_shard]`
- Added per-destination route-index tables for each source rank.
- Added a compact final compose kernel that maps each source rank's routed
  rows back to slot rows on the destination GPU.
- Avoids the source-side dense zero plus atomic reduction into the full
  `[dest][slot]` contribution layout when compact mode is enabled.
- Copies only `routes * hidden_shard` elements per source/destination instead
  of `slots * hidden_shard`.

Updated launcher/config/bench wiring.

- Added `DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE`.
- Promoted compact route-compose as the TP/EP appliance default.
- Added `--compact-route-compose` and `--no-compact-route-compose` to the
  sustained HTTP bench.
- Added the promoted default to the Kubernetes example.

## Validation

Local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-tp-ep-http-bench.sh
ruby -e 'require "yaml"; YAML.load_stream(File.read("deploy/v100/ds4-v100-appliance.k8s.yaml")); puts "yaml ok"'
kubectl apply --dry-run=client -f deploy/v100/ds4-v100-appliance.k8s.yaml
git diff --check
```

V100 validation:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Same-binary 64-token A/B at `32` slots / `256K` / three generation requests:

| Mode | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | EP ms | Compose ms | Compose copy ms | Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 711.177884 | 719.489689 | 936.910871 | 949.492675 | 2736.333652 | 3819.310485 | 1989.636922 | 96/96 |
| compact route-compose | 791.453850 | 796.894336 | 1041.001373 | 1049.369883 | 2612.143309 | 3287.921614 | 1804.154142 | 96/96 |

Same-binary uplift:

```text
wall generated tok/s:      +11.29%
wall continuation tok/s:   +10.76%
decode generated tok/s:    +11.11%
decode continuation tok/s: +10.53%
```

32-token compact sanity at `32` slots / `256K` / three generation requests:

| Tokens/request | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | Compose ms | Compose copy ms | Match |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 802.701663 | 813.475877 | 1056.056957 | 1072.771038 | 1586.147231 | 861.346646 | 96/96 |

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint284-tp-ep-compact-route-compose/
```

Subdirectories:

- `main/cluster/compact64/`
- `main/cluster/control64/`
- `compact32/cluster/compact32/`

Each contains the sustained HTTP summary, per-request responses,
`status_after.json`, `metrics.txt`, GPU utilization, and server logs.

## Decision

Promote compact route-compose as the TP/EP appliance default. It improves
same-binary 64-token serving throughput by about `11%` while preserving
aggregate `96/96` token match.

The improvement comes from avoiding the full dense source-side contribution
layout and reducing peer-copy payload for the current `top_k=6`, `slots=32`
shape. This is now the strongest post-HTTP optimization result in the TP/EP
serving path.

## Next

- Keep `DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1` enabled in serving matrices.
- Re-run the promoted 32/64 matrix after this commit as the new topline.
- Continue toward practical serving by adding request coalescing/admission,
  while preserving the compact route-compose stage metrics.
