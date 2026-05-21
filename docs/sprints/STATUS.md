# DS4 V100 Appliance Status

Last updated: 2026-05-20

## Topline

Current best 8-slot throughput is still the Sprint 111 fused TurboMind gate/up
appliance path. It turns each layer's separate gate and up expert GEMMs into one
offline-packed `gate_up` grouped GEMM and keeps the same per-GPU VRAM footprint
as the previous appliance. Sprint 114 added a guarded Volta HMMA path for the
shared-down F8 batch projection; it is correct and slightly faster in its own
same-binary run, but the gain is inside run noise and below the Sprint 113
default repeat, so it remains opt-in.

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Production fused gate_up appliance | 262,144 | 8 | `33.589285` | `31.489955` | 8/8 token match |
| Shared-down F8 HMMA opt-in | 262,144 | 8 | `33.550415` | `31.453514` | 8/8 token match |
| Direct FFN delta opt-in | 262,144 | 8 | `33.360404` | `31.275379` | 8/8 token match |
| Same-binary separate gate/up control | 262,144 | 8 | `31.312694` | `29.355651` | 8/8 token match |
| Production fused gate_up appliance | 1,048,576 | 4 | `21.403909` | `20.066165` | 4/4 token match |
| Shared-down F8 HMMA opt-in | 1,048,576 | 4 | `21.396331` | `20.059061` | 4/4 token match |
| Small-route opt-in | 1,048,576 | 4 | `20.249531` | `18.983935` | 4/4 token match |

The last pre-Sprint107 committed baseline was Sprint 104 at `31.451185`
generated tok/s for 8-slot/256K and `20.026385` for 4-slot/1M.

## Recent Experiments

| Sprint | Experiment | Result | Decision |
|---|---|---|---|
| 103 | Exact-bit E4M3 F8 decode replacing `ldexpf()` | Improved 8-slot/256K to `30.862791` generated tok/s and 4-slot/1M to `19.733742` | Shipped |
| 104 | Warp-shuffle reductions for hot F8 arena kernels | Improved 8-slot/256K repeat to `31.451185`; 4-slot/1M to `20.026385` | Shipped; current baseline |
| 105 | Extend warp reductions to BF16/F32 matmuls | Correct, but repeat result was inside Sprint 104 band | Rejected and reverted |
| 106 | Warm served `nvprof` profile of Sprint 104 | F8 rows2/grouped rows2 were ~51% GPU time; TurboMind SM70 MXFP4 was ~25%; GPU memcpy traffic was small | Use profile to choose next kernel target |
| 107 | DS4-specific grouped F8 rows2 attention-output-A kernel | Correct and faster for 8-slot/256K; neutral for 4-slot/1M | Shipped |
| 108 | TurboMind small-route count/prefix/scatter fusion | Correct; 8-slot repeat was `31.759013` opt-in vs `31.794180` rollback, while 4-slot/1M was `20.249531` opt-in vs `20.081695` rollback | Kept opt-in |
| 109 | F8 four-output-row CTA probe | Correct; regressed 8-slot/256K to `30.998275` vs `31.380225` control and 4-slot/1M to `19.898462` vs `20.041787` control | Rejected as default; opt-in only |
| 110 | TurboMind fused gate+up grouped-GEMM probe | Correct; `1.504x`, `1.532x`, and `1.462x` faster at 6, 24, and 48 total routes | Proceed to appliance implementation |
| 111 | Production fused TurboMind gate_up appliance | Correct; 8-slot/256K improved to `33.430971` from `31.312694` same-binary separate control | Shipped/default for fused packs |
| 112 | Fused appliance profile plus F8 warp-scale probe | Profile showed F8 row-pair/grouped kernels at `54.58%` GPU time; warp-scale was correct but regressed 8-slot/256K to `29.009399` vs `33.484099` control | Kept opt-in/off |
| 113 | Direct FFN delta accumulation and cached FFN input ptr table | Correct, but `33.360404` vs `33.589285` control at 8-slot/256K | Kept opt-in/off |
| 114 | DS4-shaped shared-down F8 HMMA batch kernel | Correct; `33.550415` vs `33.397763` control at 8-slot/256K and `21.396331` vs `21.365610` at 4-slot/1M | Kept opt-in/off |

## Sprint 106 Profile Takeaway

The profile does not point at disk or host RAM as the decode bottleneck.
`cudaMemcpy` API accounting is noisy, but GPU memcpy time was tiny. The main
device buckets were:

- F8 rows2 arena matmul: `38.97%`
- F8 grouped rows2 arena matmul: `12.39%`
- TurboMind SM70 MXFP4 grouped GEMM: `25.42%`

That makes the practical next targets F8 execution shape and TurboMind routed
expert scheduling.

## Current Shipped Change

Sprint 107 adds a guarded DS4-specialized CUDA kernel for the fixed grouped
attention-output-A shape:

- groups: `8`
- rows per group: `1024`
- columns per group: `4096`
- fallback: existing generic grouped rows2 kernel
- rollback knob: `DS4_V100_CUDA_F8_GROUPED_DS4_FAST=0`

Validation already completed on the cluster:

- `cuda_source_dtypes_smoke`: passed
- `cuda_v100_projection_attention_smoke`: passed
- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- `cuda_v100_selected_token_smoke --expected-token-hex 3136`: passed

Throughput completed:

- 8-slot/256K fast: `31.811137` generated tok/s, `8/8`
- 8-slot/256K fast repeat: `31.630774` generated tok/s, `8/8`
- 8-slot/256K rollback: `31.098630` generated tok/s, `8/8`
- 4-slot/1M fast: `20.095510` generated tok/s, `4/4`
- 4-slot/1M rollback: `20.105807` generated tok/s, `4/4`

Remaining optional validation:

- Focused profile to confirm whether grouped F8 kernel time moved.

## Current Opt-In Probe

Sprint 108 adds a guarded small-route TurboMind route builder:

- combines route count, prefix, and scatter into one one-block kernel for the
  production small-route shape;
- is controlled by `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD`;
- remains disabled by default because it did not improve the 8-slot/256K
  practical target.

Validation completed on the cluster:

- `cuda_v100_turbomind_adapter_smoke`: passed
- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- selected-token smoke with small-route on: passed, token id `926`, hex `3136`
- selected-token smoke with small-route off: passed, token id `926`, hex `3136`
- rebuilt default check: `turbomind_small_route_build=0`, selected-token passed

Sprint 109 adds a guarded F8 row4 CTA probe:

- computes four large F8 output rows per CTA for the ungrouped and DS4 grouped
  attention-output paths;
- is controlled by `DS4_V100_CUDA_F8_ROW4`;
- remains disabled by default because it reduced throughput in both measured
  serving tiers.

Validation completed on the cluster:

- `cuda_source_dtypes_smoke`: passed
- `cuda_v100_projection_attention_smoke`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- selected-token smoke with row4 on: passed, token id `926`, hex `3136`

Sprint 110 adds a standalone TurboMind fused gate/up benchmark:

- shape: `K=4096`, `N=2048`, fused `N=4096`, 256 experts;
- route set: six sparse active experts;
- result: fused gate_up is `1.46x-1.53x` faster than separate gate and up
  grouped calls;
- correctness: exact output match for both halves of the fused tensor.

Sprint 111 ships that fused gate/up result into the appliance:

- packer emits `blk.N.ffn_gate_up_exps.weight` with `--fuse-gate-up`;
- runtime defaults to the fused path with `DS4_V100_TURBOMIND_FUSED_GATE_UP=1`;
- selected-token smoke passed with token id `926`, hex `3136`;
- full scheduler smoke passed with `tm_layers=43`;
- 8-slot/256K served A/B improved from `31.312694` to `33.430971`
  generated tok/s;
- 4-slot/1M fused sanity passed at `21.403909` generated tok/s.

Sprint 112 profiles the fused appliance and tests a narrow F8 scale-hoist
variant:

- fused 8-slot/256K profile reached `33.972205` generated tok/s under the
  profiler harness and preserved `8/8` token matches;
- F8 row-pair plus DS4 grouped attention-output kernels were `54.58%` of GPU
  time after Sprint 111;
- warp-broadcast E8M0 scale loading passed source/projection, scheduler, and
  selected-token correctness;
- same-binary 8-slot/256K A/B regressed from `33.484099` to `29.009399`
  generated tok/s, so `DS4_V100_CUDA_F8_WARP_SCALE=0` remains the default.

Sprint 113 tests direct FFN delta accumulation:

- batch scratch now exposes contiguous FFN norm/delta tensors with stable
  per-slot views;
- TurboMind routed FFN wrappers can consume an existing device pointer table;
- TurboMind routed FFN wrappers can accumulate into an existing output tensor;
- selected-token correctness passed with `DS4_V100_FFN_DIRECT_DELTA=1`;
- same-binary 8-slot/256K A/B was `33.360404` generated tok/s with direct delta
  versus `33.589285` control, so `DS4_V100_FFN_DIRECT_DELTA=0` remains the
  default.

Sprint 114 tests a DS4-shaped shared-down F8 HMMA batch kernel:

- the kernel is guarded by `DS4_CUDA_F8_HMMA_SHARED_DOWN=1`;
- it only dispatches for `rows=4096`, `cols=2048`, and `n_tokens=4/8`;
- focused target-shape smoke passed against the existing scalar F8 path;
- full scheduler and selected-token smokes passed with the fused appliance;
- same-binary A/B showed small positive deltas, but not enough to promote:
  `33.550415` vs `33.397763` at 8-slot/256K and `21.396331` vs `21.365610`
  at 4-slot/1M.

## Next Target

The next target needs to change a larger execution boundary. Fused gate/up
removed one large grouped GEMM launch per routed layer, but aggregate throughput
is still only about `33.6` tok/s, far below the practical serving target. The
Sprint 114 HMMA probe shows that a naive small-M WMMA tile is not enough by
itself. The next sprint should either fuse a larger F8 FFN region
(gate/up/SwiGLU/down) or deepen TurboMind expert scheduling so the kernel owns
more work per launch and raises tensor-core occupancy.

The concise current status is also tracked in
`docs/sprints/EXPERIMENT-STATUS.md`.
