# Sprint 389: Promote Compressed Dense Stats Skip

## Overview

Revalidate and, if justified, promote the existing TP/EP gate that skips
host-side dense-output statistics in the true DS4 compressed-KV projection
path.

Sprint 372 showed this gate was a large serving candidate: at `32` slots /
`256K`, it preserved selected-token parity, improved direct scaffold decode
from `100.739521` to `117.463961` tok/s, improved full chat server decode from
`99.748339` to `117.340768` tok/s, and improved client throughput from
`51.345855` to `58.923892` tok/s. It stayed default-off because normal chat
token/text parity needed a stronger deterministic check.

Sprints 384-388 shifted the current default baseline to real model-router
routing plus compact MoE. This sprint repeats the gate against that current
baseline and makes a promote/reject decision.

## Scope

- Use only the TP/EP path. No PP/layer-split work.
- Use the existing default-off gate:
  `--true-ds4-compressed-kv-skip-dense-stats-gate`.
- Validate at the target appliance shape:
  `32` slots, `32` active requests, `256K` context, `position=262080`.
- Run both:
  - direct token-major A/B for first-token/checksum/stage timing
  - HTTP `/v1/chat/completions` A/B for serving throughput and token parity
- Keep VRAM admission enabled with `--vram-report --vram-min-free-mib 64`.
- If the candidate preserves parity and improves serving metrics, promote the
  launcher default:
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1`.

## Out Of Scope

- No CUDA graph work.
- No MTP integration.
- No TP-sharded expert integration.
- No route-planner rewrite.
- No dtype conversion or new kernel family.

## Definition Of Done

- Sprint document exists before execution.
- V100 direct same-binary A/B completes for control and candidate.
- V100 HTTP same-binary A/B completes for control and candidate.
- First token and available deterministic response/token evidence match.
- Decode/server/client throughput and compressed-KV timing are recorded.
- Gate is either promoted as default or explicitly rejected/deferred.
- `docs/sprints/VISION.md`, `docs/sprints/STATUS.md`, and
  `TEMP_STATUS_REPORT_389.md` are updated.
- Kept artifacts are committed.

## Risks

- Skipping dense stats removes diagnostic visibility; if parity fails or
  errors become less visible, the gate must remain opt-in.
- HTTP client throughput can be noisy, so promotion should depend on parity
  plus a clear server-side decode/stage win, not a single marginal client
  number.
- If current real-router changes altered the compressed-KV timing profile,
  Sprint 372's gain may not reproduce.

## Execution Plan

1. Run direct token-major control with model-router routes, compact MoE, and
   VRAM admission.
2. Run direct token-major candidate with the same flags plus
   `--skip-compressed-dense-stats`.
3. Run HTTP chat control with `32` requests, `32` slots, `32` generated tokens,
   model-router routes, compact MoE, and VRAM admission.
4. Run HTTP chat candidate with the same flags plus
   `--skip-compressed-dense-stats`.
5. Compare first token, checksum/token evidence, server decode tok/s, client
   generated tok/s, compressed-KV sum, and GPU utilization.
6. Promote, reject, or defer with a concrete reason.

## Outcome

Complete. The skip-dense-stats gate is promoted as the TP/EP launcher default.

Code change:

- `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS` now defaults to `1`
  in `tools/ds4-v100-run-appliance.sh`.
- The existing explicit opt-out remains:
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=0`.
- `tools/ds4-v100-tp-ep-profile.py` now matches the promoted production
  default and exposes `--disable-skip-compressed-dense-stats` for future
  control runs.

Direct token-major A/B at `32` slots / `256K` / `position=262080` /
`32` decode steps, with model-router routes and compact MoE:

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

HTTP `/v1/chat/completions` A/B at `32` requests / `32` slots / `256K` /
`position=262080` / `32` generated tokens/request:

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

Parity check:

- Direct first token matched.
- HTTP first token matched.
- All `32` HTTP responses had identical generated token sequences.
- All `32` HTTP responses had the same response checksum:
  `17913667583206000416`.
- The raw semantic comparator reported `4/32` diffs only because cache/request
  metadata differed; generated text and token IDs matched.

Launcher proof on gpu-01:

```text
unset/default: --true-ds4-compressed-kv-skip-dense-stats-gate present
explicit 0:   gate absent
```

## Decision

Promote. This removes production-host diagnostic copies/synchronization from
the compressed/indexer dense projection path, preserves generated tokens and
checksums, improves direct decode by `11.97%`, improves HTTP server decode by
`15.66%`, improves HTTP client throughput by `5.71%`, and leaves the existing
explicit opt-out available.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint389-skip-dense-stats/direct-control`
  - `/workspace/logs/sprint389-skip-dense-stats/direct-candidate`
  - `/workspace/logs/sprint389-skip-dense-stats/http-control`
  - `/workspace/logs/sprint389-skip-dense-stats/http-candidate`
- Local:
  - `logs/from-cluster/sprint389-skip-dense-stats`
