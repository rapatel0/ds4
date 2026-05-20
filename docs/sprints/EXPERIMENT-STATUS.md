# DS4 V100 Experiment Status

Last updated: 2026-05-20

## Topline

The appliance is correct and served on the 8x V100 node, but it is not yet in
the practical throughput range from the vision. Current measured decode
throughput is still about `32` aggregate tok/s at the 8-slot/256K target.

| Track | Context | Slots | Best Generated tok/s | Current Default Generated tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Throughput serving target | 262,144 | 8 | `31.811137` | `31.794180` | 8/8 token match |
| Long-context target | 1,048,576 | 4 | `20.249531` opt-in | `20.081695` | 4/4 token match |

The `20.249531` long-context result uses the Sprint 108 small-route build path,
but that path is not the default because the 8-slot/256K A/B was neutral to
slightly worse.

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

## Remaining

- Close the throughput gap. The current `~32` tok/s aggregate is far below the
  `~1k-2k` practical target discussed in the vision.
- Improve GPU utilization. The latest profile says the bottleneck is device
  kernel shape/occupancy, not disk, host RAM, or bulk PCIe/NVLink traffic.
- Attack larger hot-path buckets instead of small host-side route plumbing:
  - F8 arena rows2 / grouped rows2 execution shape.
  - TurboMind MXFP4 expert occupancy and route-expanded activation layout.
  - Fusing or batching a larger layer boundary than the isolated projection and
    route-build attempts.
- Decide whether the next production step is a deeper TurboMind adapter change
  or an F8 matmul tiling/vectorization change, based on a fresh profile after
  Sprint 107.

## Operator Status

The default launcher now keeps `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0`. The
opt-in diagnostic path can be enabled with:

```text
DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=1
```

The current default selected-token smoke passed after rebuilding with the
opt-in path disabled by default.
