# Sprint 148 - SM70 Stage-4 Gate/Up Software-Pipeline Probe

Date: 2026-05-21

## Objective

Test whether a deeper SM70 software pipeline in the fused MXFP4
gate/up+gated-SiLU kernel produces a material end-to-end gain on the V100
appliance.

## Changes

- Added explicit stage-4 (`s4`) TurboMind probe variants:
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m64_s4`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_768_m128_s4`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_1536_m64_s4`
  - `ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128_s4`
- Exposed runtime selection through explicit probe modes such as `m128_s4`,
  `m64_s4`, and `m128_s4_1536`.
- Extended the standalone TurboMind gate/up benchmark to select the new probe
  variants.
- Updated the appliance launcher whitelist so explicit stage-4 probes can be
  tested through the normal served path.
- Left `auto` unchanged.

## Standalone Results

Compact routed benchmark, 6 active groups:

| Shape | Baseline/probe | Probe ms | Comparison |
|---|---|---:|---:|
| 768 routes | `m128` | `0.6033` | baseline |
| 768 routes | `m128_s4` | `0.5811` | `1.038x` faster than `m128` |
| 768 routes | `m64_s4` | `0.6529` | slower |
| 1536 routes | `m128_1536` | `0.8694` | baseline |
| 1536 routes | `m128_s4_1536` | `0.8671` | neutral |
| 1536 routes | `m64_s4_1536` | `1.0868` | slower |

The only isolated positive result was `m128_s4` at the 768-route shape.

## Scheduler Validation

Full 43-layer scheduler smoke passed for `m128_s4`:

```text
ctx=32768
slots=128
tm_layers=43
token=16
result=ok
```

## Served A/B

128-slot / 32K served soak, 16 generated tokens, 128 requests:

| Mode | Generated tok/s | Prompt tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---|
| control `auto` | `59.865668` | `67.348876` | `56.124063` | 128/128 |
| `m128_s4` | `60.049057` | `67.555189` | `56.295991` | 128/128 |

The served delta was only about `+0.3%`, inside the run band for this tier.

## Profile Result

The full-scheduler profile did not show a reliable hot-bucket reduction.
Control gate/up share was roughly `57-60%` of routed-FFN profile time; the
`m128_s4` profile remained roughly `58-61%`, with several stages neutral or
slower.

## Artifacts

- `logs/from-cluster/sprint148-gate-up-s4-standalone/`
- `logs/from-cluster/sprint148-smoke-gate-m128-s4-128slot/`
- `logs/from-cluster/sprint148-served-128slot-32k-control-auto/`
- `logs/from-cluster/sprint148-served-128slot-32k-candidate-m128-s4/`
- `logs/from-cluster/sprint148-profile-auto-128slot-32k/`
- `logs/from-cluster/sprint148-profile-m128-s4-128slot-32k/`

## Decision

Keep stage-4 probes explicit opt-in only. This is a real software-pipeline
experiment on the fused kernel, but it does not move end-to-end throughput
materially. The next performance sprint should stop varying only the existing
TurboMind mainloop stage count and instead prototype a larger routed-FFN
executor boundary or a TP/EP microbenchmark.
