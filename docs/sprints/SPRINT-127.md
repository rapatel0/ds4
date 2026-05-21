# Sprint 127 - TurboMind Gated-SiLU Interleaved Pack

Date: 2026-05-21

## Objective

Turn the Sprint 126 routed-expert profile into one bounded production-path
fusion: use TurboMind's gated-SiLU epilogue for DS4 routed gate/up experts by
packing fused expert rows as `[gate0, up0, gate1, up1, ...]`.

This tests whether a CUTLASS/TurboMind-style epilogue fusion and a better
offline format are enough to move the served topline before committing to a
larger persistent routed-expert kernel.

## Implementation

- Added optional TurboMind ABI:
  - `ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens`
  - same grouped total-token route contract as the fused gate/up path
  - requires interleaved fused rows and emits `[total_routes, mid]` half output
- Extended appliance pack metadata with a tensor flag:
  - `DS4_GPU_TURBOMIND_MXFP4_GATE_UP_INTERLEAVED`
- Added `--fuse-gate-up-interleaved` to the appliance packer:
  - implies fused gate/up packing
  - emits runtime layout `turbomind_mxfp4_grouped_gate_up_interleaved`
  - emits kernel family `turbomind_mxfp4_grouped_gated_silu_sm70`
- Added guarded runtime dispatch:
  - `DS4_V100_TURBOMIND_GATED_SILU=1`
  - requires the new TurboMind ABI and interleaved pack metadata
  - fails closed if an interleaved pack is loaded without the gated path
- Moved route-weight application for the gated path to the post-down
  scatter/reduce step, since the gated epilogue does not consume DS4 route
  weights directly.

Defaults remain unchanged. Existing Sprint 111 fused packs continue to use the
existing `[all gate][all up]` path unless the new gated flag and interleaved
pack are both present.

## Validation

Local/static:

```text
bash -n tools/ds4-v100-run-appliance.sh tools/ds4-v100-appliance-soak.sh
git diff --check
```

Cluster build on `llamacpp-build-8gpu` under `/workspace/ds4`:

```text
cmake --build build/turbomind-v100-s127 \
  --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80

CUDA_ARCH=sm_70 make -j80 \
  tools/ds4-v100-appliance-pack \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

TurboMind gated microbench:

```text
test_ggml_turbomind_grouped_gate_up_fusion \
  build/turbomind-v100-s127/libggml-turbomind.so
```

Full interleaved appliance pack:

```text
tools/ds4-v100-appliance-pack \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /workspace/ds4-appliance-full-tm-gated-s127 \
  --pack-gpu 0 \
  --fuse-gate-up-interleaved \
  --lib build/turbomind-v100-s127/libggml-turbomind.so
```

Scheduler validation:

```text
DS4_V100_TURBOMIND_GATED_SILU=1 \
tests/cuda_v100_stage_scheduler_smoke --stage 0 --slots 16 --ctx 262144

DS4_V100_TURBOMIND_GATED_SILU=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43

DS4_V100_TURBOMIND_GATED_SILU=1 \
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1 \
tests/cuda_v100_full_scheduler_smoke --slots 16 --ctx 262144 --expect-tm-layers 43
```

Served A/B:

```text
tools/ds4-v100-appliance-soak.sh \
  --ctx 262144 --slots 16 --active-microbatch 16 \
  --tokens 16 --requests 16 --warmup-requests 1
```

## Results

TurboMind standalone grouped gate/up test:

| Total routes shape | Separate gate/up | Fused gate/up | Gated-SiLU | Gated speedup |
|---:|---:|---:|---:|---:|
| 6 routes | `0.2511 ms` | `0.1696 ms` | `0.1673 ms` | `1.500x` |
| 24 routes | `0.2343 ms` | `0.1552 ms` | `0.1509 ms` | `1.552x` |
| 48 routes | `0.2133 ms` | `0.1444 ms` | `0.1447 ms` | `1.474x` |

The gated path compared against the separate half-rounded reference with
relative error around `2.9e-4`. The max absolute delta is expected to be larger
than the fused gate/up parity test because the gated epilogue changes rounding:
it applies SiLU before the separate half materialization that the old reference
uses.

Full interleaved appliance pack:

```text
source_rows=1199
tm_rows=86
skipped_rows=43
source_bytes=8973123932
tm_weight_bytes=138512695296
tm_scale_bytes=8657043456
```

Full 43-layer smoke:

```text
layers=43
tm_layers=43
uploaded_tensors=8
uploaded_bytes=156142896212
ok
```

Served A/B at `ctx=262144`, `slots=16`, `active_microbatch=16`:

| Appliance | Gated-SiLU | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|
| Sprint 111 fused gate/up control | off | `43.691032` | `40.960343` | `16/16` token match |
| Sprint 127 interleaved gated candidate | on | `43.933293` | `41.187462` | `16/16` token match |

Profiled full 43-layer routed-FFN aggregate with the gated path:

| Stage | Time | Share of profiled routed-FFN time |
|---|---:|---:|
| route build | `4.990 ms` | `18.7%` |
| activation gather | `1.031 ms` | `3.9%` |
| gated gate/up grouped GEMM | `12.122 ms` | `45.3%` |
| standalone SwiGLU | `0.000 ms` | `0.0%` |
| down grouped GEMM | `6.992 ms` | `26.2%` |
| scatter/reduce | `1.228 ms` | `4.6%` |
| total | `26.734 ms` | `100%` |

Compared with Sprint 126's `28.242 ms` profiled routed-FFN total, the gated
path removes the standalone SwiGLU bucket and lowers the profiled routed path
by about `5.3%`. The served topline improves by only about `0.55%`, which is
inside the current end-to-end bottleneck mix.

## Decision

Keep the gated-SiLU/interleaved-pack path as a correct opt-in path, but do not
promote it as the production default yet.

This sprint confirms the format and kernel-direction thesis:

- pack layout matters;
- epilogue fusion can remove a real intermediate and launch;
- TurboMind/CUTLASS-style software-pipelined kernels are the right source of
  ideas for V100;
- small launch/epilogue fusions are not enough to close the throughput gap.

The next production optimization should be a larger persistent routed-expert
pipeline that combines route-expanded gather, packed MXFP4 dequant staging,
gate/up HMMA, gated activation, down HMMA, and weighted scatter/reduce across
the same route batch. That is the likely path to materially higher GPU
utilization.
