# Sprint 080 Report: Copied tc-grid V100 INT8 Kernel Proof

## Outcome

`SHIP_PROOF_ONLY`.

Sprint 080 copied the relevant tc-grid V100 source into `ds4`, added a
repository-local CUDA smoke/bench, and proved the copied `v13_rf_v6` INT8 HMMA
kernel on the V100 pod. This is useful infrastructure and performance evidence,
but it should not become the default model path yet. The DS4 routed experts are
source MXFP4, and the measured INT8 path only reaches strong utilization once
effective `M` is large.

TurboMind remains the better next candidate for the production routed expert
path because it has stronger evidence for V100 tensor-core utilization and
matches the DS4 MXFP4 expert layout more directly.

## Changes

- Copied tc-grid source into `kernels/tc-grid/`:
  - `include/tc_grid.h`
  - `include/dispatch.h`
  - `kernels/mma_sm70.cuh`
  - `kernels/v12_kernels.cuh`
  - `kernels/v13_kernels.cuh`
  - `LICENSE.turbomind`
  - `README.md`
- Added `tests/cuda_v100_tc_grid_int8_smoke.cu`.
- Added a Makefile target for `tests/cuda_v100_tc_grid_int8_smoke`.
- Updated `docs/architecture/DS4-V100-LAYOUT.md` with the copied low-bit kernel
  policy.

## Validation

Local:

```bash
git diff --check
```

V100 build:

```bash
CUDA_ARCH=sm_70 make tests/cuda_v100_tc_grid_int8_smoke
```

The V100 build completed. The only compiler note was an unused constant warning
inside the copied `v13_kernels.cuh`.

V100 correctness:

```bash
./tests/cuda_v100_tc_grid_int8_smoke
```

Result:

```text
tc_grid_int8_correctness M=128 N=128 K=128 max_abs=0 p99_abs=0 mean_abs=0 ok
```

## Timing Evidence

Node was idle at measurement start: all eight V100s reported `0 MiB` memory and
`0%` GPU utilization.

The default smoke shape is DS4 expert-like for a large enough routed batch:
`M=128, N=2048, K=4096`.

| Shape | Mean ms | TFLOP/s | GB/s | Result |
|---|---:|---:|---:|---|
| `M=128 N=2048 K=4096` | `0.297329` | `7.223` | `40.557` | ok |
| `M=256 N=2048 K=4096` | `0.294666` | `14.576` | `51.599` | ok |
| `M=512 N=2048 K=4096` | `0.299848` | `28.648` | `71.689` | ok |
| `M=1024 N=2048 K=4096` | `0.428657` | `40.078` | `79.501` | ok |
| `M=2048 N=2048 K=4096` | `0.848108` | `40.513` | `69.855` | ok |
| `M=2048 N=7168 K=7168` | `4.536473` | `46.391` | `37.922` | ok |

## Decision

Keep copied tc-grid as a proven local kernel source and benchmark harness, but
do not wire INT8 into the DS4 runtime yet.

The result confirms the kernel family can use V100 tensor cores well when
effective `M` is high, but it also confirms the current practical serving
problem: low-M routed decode does not naturally feed this shape. Converting
source MXFP4 experts to INT8 would also expand expert memory, so it needs a
separate planner and quality gate before runtime use.

Next sprint should copy and prove the TurboMind MXFP4 grouped GEMM path from
inside `ds4`, or directly adapt the DS4 routed path to a TurboMind-shaped
grouped execution plan if the copied build surface is manageable.

## Artifacts

- `logs/from-cluster/sprint080-tcgrid-v100.log`
