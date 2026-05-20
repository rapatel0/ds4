# Sprint 111 - Production TurboMind Fused Gate/Up

Date: 2026-05-20

## Objective

Move the Sprint 110 fused gate/up TurboMind result into the production appliance
path without breaking existing appliances.

## Plan

Add an optional fused routed expert tensor:

```text
blk.N.ffn_gate_up_exps.weight
```

The offline appliance packer should create it by concatenating gate expert rows
then up expert rows per expert and packing the result as one TurboMind MXFP4
matrix with `N = 2 * intermediate`.

Runtime behavior:

- If the fused tensor is present and `DS4_V100_TURBOMIND_FUSED_GATE_UP=1`, use
  one grouped TurboMind GEMM for gate+up.
- If the fused tensor is missing or the knob is off, keep the existing separate
  gate and up grouped calls.
- Preserve the existing down grouped GEMM, route sorting, route weights, and
  scatter-sum semantics.

## Definition of Done

- [x] Packer can emit fused gate_up tensors with `--fuse-gate-up`.
- [x] Existing non-fused appliances still bind and run.
- [x] Fused appliances bind through `ds4_v100_context` and
  `ds4_v100_layer_state`.
- [x] CUDA routed FFN has a fused gate_up execution path.
- [x] V100 correctness passes selected-token and scheduler smokes.
- [x] Served A/B throughput is recorded before defaulting the fused path.

## Implementation

- Added `--fuse-gate-up` to `tools/ds4-v100-appliance-pack.cu`.
- Added `--keep-separate-gate-up` for bounded rollback diagnostics.
- Added `blk.N.ffn_gate_up_exps.weight` as a TurboMind routed expert tensor.
- Added `DS4_V100_TURBOMIND_FUSED_GATE_UP`, default enabled.
- Added fused single-token and batch routed FFN CUDA entry points.
- Extended `tests/v100_layer_state_smoke` to validate TurboMind fused metadata.

The full fused production appliance was generated at:

```text
/workspace/ds4-appliance-full-tm-fused-s111
```

Pack summary:

```text
source_rows=1199
tm_rows=86
skipped_rows=43
tm_weight_bytes=138512695296
tm_scale_bytes=8657043456
gpu0.weights bytes=22524134668
gpu1.weights bytes=21494393612
gpu2.weights bytes=21494393612
gpu3.weights bytes=21494393612
gpu4.weights bytes=21494393612
gpu5.weights bytes=17922654732
gpu6.weights bytes=17901334540
gpu7.weights bytes=11817197824
```

The per-GPU shard sizes match the previous separate gate/up appliance. The
TurboMind row count drops from 129 to 86 because each layer now has `down` plus
fused `gate_up`, instead of `down`, `gate`, and `up`.

## Validation

Cluster: `llamacpp-build-8gpu` on `gpu-01`.

Build:

```text
CUDA_ARCH=sm_70 make -j80 \
  tools/ds4-v100-appliance-pack \
  tools/ds4-v100-replay \
  tests/v100_layer_state_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tests/cuda_v100_selected_token_smoke
```

Correctness:

- Bounded keep-separate layer-state smoke: passed for layer 0.
- Full fused layer-state smokes: passed for layers 0, 12, 35, and 42.
- Full scheduler smoke: passed with `tm_layers=43`.
- Selected-token smoke: passed, token id `926`, expected text hex `3136`.

## Throughput

Served A/B, same binary, `ctx=262144`, `slots=8`,
`active_microbatch=8`, `tokens=16`, `requests=8`,
`async_pipeline_mode=per-step`.

| Appliance | Generated tok/s | Continuation tok/s | Avg latency ms | Correctness |
|---|---:|---:|---:|---|
| Separate gate/up `/workspace/ds4-appliance-full-tm-s090` | `31.312694` | `29.355651` | `4040.542` | 8/8 |
| Fused gate_up `/workspace/ds4-appliance-full-tm-fused-s111` | `33.430971` | `31.341535` | `3780.481` | 8/8 |

This is a `6.76%` generated-throughput improvement and a `6.77%`
continuation-throughput improvement for the main 8-slot/256K target.

Long-context fused sanity, `ctx=1048576`, `slots=4`, `active_microbatch=4`,
`tokens=16`, `requests=4`, `async_pipeline_mode=per-step`:

```text
generated tok/s:     21.403909
continuation tok/s:  20.066165
correctness:         4/4 token match
```

## Decision

Ship fused gate/up as the production appliance default when the appliance pack
contains fused tensors. Use `DS4_V100_TURBOMIND_FUSED_GATE_UP=0` only with an
appliance that also contains separate gate/up tensors.

## Risks

- Fused-only appliances do not contain separate gate/up tensors, so runtime
  fallback requires either an old appliance or a packer mode that keeps both.
- Packing both separate and fused gate/up increases VRAM pressure; it should be
  used only for bounded A/B diagnostics unless memory telemetry proves it fits.
- The fused output layout is `[route][gate_mid, up_mid]`, so the existing
  contiguous two-buffer SwiGLU kernel cannot be reused directly.

## Artifacts

- `tools/ds4-v100-appliance-pack.cu`
- `ds4_v100_context.c`
- `ds4_v100_layer_state.{c,h}`
- `ds4_gpu.h`
- `ds4_cuda.cu`
- `tools/ds4-v100-run-appliance.sh`
- `deploy/v100/ds4-v100-appliance.env.example`
- `logs/from-cluster/sprint111-fused-gate-up/`
