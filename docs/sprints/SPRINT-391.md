# Sprint 391: Longer E5M2 KV Parity A/B

## Overview

Re-test the default-off E5M2 typed-KV gate at the real `32` slot / `256K`
serving shape with the promoted Sprint 389 defaults and the permanent Sprint
390 HTTP parity comparator.

Sprint 381 showed E5M2 KV was row-correct and promising in a short 4-token
run, but it stayed default-off because E5M2 gives up mantissa precision and
only short selected-token parity was proven. This sprint performs a longer
decode-heavy A/B against the current real-router compact-MoE baseline.

## Scope

- Use only the TP/EP path. No PP/layer-split work.
- Use the existing default-off gate:
  `--fp8-e5m2-kv`.
- Keep current promoted defaults, including skip compressed dense stats.
- Validate at:
  `32` slots, `32` active requests, `256K` context, `position=262080`,
  `32` generated tokens/request.
- Run direct token-major A/B and HTTP chat A/B.
- Compare HTTP responses with
  `tools/ds4-v100-http-response-parity.py`.

## Out Of Scope

- No new KV format implementation.
- No MTP work.
- No TP-sharded expert work.
- No router planner changes.

## Definition Of Done

- Sprint document exists before execution.
- V100 direct control/candidate A/B completes.
- V100 HTTP control/candidate A/B completes.
- HTTP parity comparator summary is written.
- Decision is documented: promote, reject, or keep diagnostic-only with a
  concrete next condition.
- Status/vision/report are updated and committed.

## Risks

- E5M2 may preserve first token but diverge later in the generated sequence.
- E5M2 may improve stage timing but fail response parity, which would make it
  unsuitable as a quality-preserving default.
- Prior candidate startup once hit OOM; keep VRAM admission enabled.

## Execution Plan

1. Run direct token-major control with current defaults.
2. Run direct token-major candidate with `--fp8-e5m2-kv`.
3. Run HTTP chat control with current defaults.
4. Run HTTP chat candidate with `--fp8-e5m2-kv`.
5. Compare HTTP response directories using the permanent parity comparator.
6. Decide promotion status from parity, VRAM admission, and throughput.

## Outcome

Complete. E5M2 KV remains diagnostic-only.

During execution, the first direct control run exposed a profiler bug: HTTP
mode inherited the promoted skip-dense-stats default, but direct-token-major
mode only emitted `--true-ds4-compressed-kv-skip-dense-stats-gate` when
`--skip-compressed-dense-stats` was explicit. This sprint fixed direct-mode
command generation so direct profiles now match the promoted default unless
`--disable-skip-compressed-dense-stats` is set.

## Results

Direct token-major A/B at `32` slots / `256K` / `position=262080` /
`32` decode steps:

| Metric | Control | E5M2 KV |
|---|---:|---:|
| First token | `98751` | `98751` |
| Generated decode tok/s | `103.237368` | `102.152512` |
| Generated wall tok/s | `65.559769` | `69.125275` |
| Compressed-KV sum | `1808.998742 ms` | `1799.161576 ms` |
| Pre-EP compressed-KV | `1821.203311 ms` | `1812.406574 ms` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

HTTP `/v1/chat/completions` A/B at `32` requests / `32` slots / `256K` /
`position=262080` / `32` generated tokens/request:

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

Parity:

```text
match=true
paired_count=32
matched_pairs=32
failed_pairs=0
```

## Decision

Keep E5M2 KV default-off. The HTTP serving result is positive and exact-token
parity passed for this 32-response run, but direct decode was slightly slower
and E5M2 still carries a precision-risk profile. Promotion now requires a
broader multi-prompt parity/soak run, not just one repeated prompt template.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint391-e5m2-kv/direct-control-default`
  - `/workspace/logs/sprint391-e5m2-kv/direct-candidate-e5m2`
  - `/workspace/logs/sprint391-e5m2-kv/http-control-default`
  - `/workspace/logs/sprint391-e5m2-kv/http-candidate-e5m2`
- Local:
  - `logs/from-cluster/sprint391-e5m2-kv`
