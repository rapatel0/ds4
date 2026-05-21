# DS4 V100 Experiment Status

Last updated: 2026-05-20

## Topline

The appliance is correct and served on the 8x V100 node, but it is not yet in
the practical throughput range from the vision. The current default is the
Sprint 111 fused TurboMind gate/up appliance path; Sprint 113 repeated that
default at `33.589285` generated tok/s in the same binary used for the rejected
direct FFN-delta probe.

| Track | Context | Slots | Best Generated tok/s | Current Default Generated tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Throughput serving target | 262,144 | 8 | `33.589285` | `33.589285` | 8/8 token match |
| Long-context target | 1,048,576 | 4 | `21.403909` | `21.403909` | 4/4 token match |

The older `20.249531` long-context result used the Sprint 108 small-route build
path, but that path is not the default because the 8-slot/256K A/B was neutral
to slightly worse.

## Tested

- Full 8-GPU resident appliance pack on `gpu-01` using k8s-local `/workspace`.
- Full 43-layer scheduler over the TurboMind appliance pack.
- Selected-token oracle for the official short prompt, expected text hex
  `3136`, selected token id `926`.
- HTTP served soak benchmarks at:
  - `ctx=262144`, `slots=8`, `active_microbatch=8`, 16 generated tokens.
  - `ctx=1048576`, `slots=4`, `active_microbatch=4`, 16 generated tokens.
- MTP exact commit path was previously validated, but it is not the current
  throughput default because exact verification did not improve tok/s.
- Copied TurboMind MXFP4 kernels are in the production appliance path for
  routed experts; copied tc-grid INT8 kernels remain proof artifacts, not the
  selected source-quality path.

## Recent Experiment Results

| Sprint | Change | Result | Decision |
|---|---|---|---|
| 103 | Exact-bit E4M3 F8 decode | Raised 8-slot/256K to `30.862791` | Shipped |
| 104 | Warp reductions for F8 arena kernels | Raised 8-slot/256K repeat to `31.451185` | Shipped |
| 105 | BF16/F32 warp reductions | Correct but no gain | Rejected |
| 106 | Warm served `nvprof` profile | F8 rows2/grouped rows2 ~51% GPU time; TurboMind ~25% | Used for targeting |
| 107 | DS4 grouped F8 attention-output kernel | Best 8-slot/256K `31.811137` | Shipped/default |
| 108 | TurboMind small-route build fusion | Correct; `31.759013` opt-in vs `31.794180` rollback on repeat | Kept opt-in |
| 109 | F8 row4 CTA probe | Correct; `30.998275` row4 vs `31.380225` control at 8-slot/256K | Rejected as default |
| 110 | TurboMind fused gate/up grouped-GEMM probe | Correct; fused gate_up was `1.46x-1.53x` faster than separate gate and up calls | Proceed to appliance implementation |
| 111 | Production fused TurboMind gate_up appliance | Correct; `33.430971` fused vs `31.312694` same-binary separate control at 8-slot/256K; `21.403909` at 4-slot/1M | Shipped/default for fused packs |
| 112 | Fused appliance profile and F8 warp-scale probe | F8 row-pair/grouped kernels were `54.58%` GPU time; warp-scale was correct but `29.009399` vs `33.484099` control at 8-slot/256K | Kept opt-in/off |
| 113 | Direct FFN delta accumulation | Correct; `33.360404` direct delta vs `33.589285` control at 8-slot/256K | Kept opt-in/off |

## Remaining

- Close the throughput gap. The current `~32` tok/s aggregate is far below the
  `~1k-2k` practical target discussed in the vision.
- Improve GPU utilization. The latest profile says the bottleneck is device
  kernel shape/occupancy, not disk, host RAM, or bulk PCIe/NVLink traffic.
- Attack larger hot-path buckets instead of small host-side route plumbing:
  - TurboMind MXFP4 expert occupancy and route-expanded activation layout.
  - Persistent/grouped expert execution beyond the shipped Sprint 111 fused
    gate_up launch reduction.
  - A real tiled/persistent F8 projection rewrite. The Sprint 112 warp-scale
    probe and Sprint 113 direct-delta probe show that very small scalar or
    host-boundary changes can regress.
- Decide whether the next production step is a deeper TurboMind adapter change
  or a lower-level CUTLASS/TurboMind-inspired persistent kernel probe.

## Operator Status

The default launcher now keeps `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0`,
`DS4_V100_CUDA_F8_ROW4=0`, `DS4_V100_CUDA_F8_WARP_SCALE=0`, and
`DS4_V100_FFN_DIRECT_DELTA=0`. The opt-in diagnostic paths can be enabled with:

```text
DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=1
DS4_V100_CUDA_F8_ROW4=1
DS4_V100_CUDA_F8_WARP_SCALE=1
DS4_V100_FFN_DIRECT_DELTA=1
DS4_V100_TURBOMIND_FUSED_GATE_UP=0
```

The fused gate/up path is default-enabled for appliances that contain fused
`ffn_gate_up_exps.weight` tensors. Set `DS4_V100_TURBOMIND_FUSED_GATE_UP=0`
only when the appliance also contains separate gate/up tensors.
