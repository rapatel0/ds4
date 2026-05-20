# Sprint 100 - TurboMind Sync Readback A/B

Date: 2026-05-20

## Objective

Reduce hot-path synchronization around packed TurboMind routed expert GEMMs
without changing the DS4 appliance format or source-model math.

## Changes

- Added an optional TurboMind C ABI:
  - `ggml_turbomind_mul_mat_grouped_total_tokens()`
- Wired the DS4 CUDA wrapper to use that ABI when
  `DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=0`.
- Made packed TurboMind route validation readback debug-only:
  - production default: `DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=0`
  - debug rollback: `DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=1`
- Kept the older TurboMind row-count readback as the production default after
  V100 A/B showed it faster:
  - production default: `DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1`
- Documented the new knobs in launcher, k8s env, and appliance operations docs.

The important implementation detail is that the new ABI is opt-in. It removes
the `expert_offsets[num_experts]` device-to-host readback inside TurboMind, but
on the current scheduler the wait mostly moves to existing device
synchronization points instead of improving wall-clock throughput.

## Validation

Cluster build:

```text
make -C /workspace/ds4-sprint082 tools/ds4-v100-replay CUDA_ARCH=sm_70 -j8
```

TurboMind export and microtest:

```text
nm -D libggml-turbomind.so | grep ggml_turbomind_mul_mat_grouped
./build/turbomind-v100/test_ggml_turbomind_grouped_compare ./build/turbomind-v100/libggml-turbomind.so
```

The new symbol is exported and the grouped compare test passes for gate/up and
down DS4 shapes.

Same-binary appliance soaks:

| Scenario | ABI / route validation | Generated tok/s | Continuation tok/s | Correctness |
| --- | --- | ---: | ---: | --- |
| 4 slots, 1M ctx | new ABI, route sync off | `17.470095` | `16.378214` | `token_match=4/4` |
| 4 slots, 1M ctx | old ABI, route sync on | `17.796721` | `16.684426` | `token_match=4/4` |
| 8 slots, 256K ctx | new ABI, route sync off | `23.203732` | `21.753499` | `token_match=8/8` |
| 8 slots, 256K ctx | new ABI, route sync off, repeat | `23.290369` | `21.834721` | `token_match=8/8` |
| 8 slots, 256K ctx | old ABI, route sync on | `26.111551` | `24.479579` | `token_match=8/8` |
| 8 slots, 256K ctx | old ABI, route sync on, repeat | `26.159147` | `24.524200` | `token_match=8/8` |
| 8 slots, 256K ctx | old ABI, route sync off | `26.422424` | `24.771023` | `token_match=8/8` |
| 8 slots, 256K ctx | production default after flip | `26.372672` | `24.724380` | `token_match=8/8` |

Served 4-slot, 1M-context `nvprof` request-window comparison:

| Path | Key API profile | GPU copy profile |
| --- | --- | --- |
| production default, old ABI + route sync off | `cudaMemcpy 857.04 ms / 2376 calls`, `cudaDeviceSynchronize 25.22 ms / 165 calls` | DtoH `4.79 ms / 2607 calls` |
| new ABI + route sync off | no top-level `cudaMemcpy` API bucket; `cudaDeviceSynchronize 760.65 ms / 165 calls` | DtoH `1.34 ms / 231 calls` |
| old ABI + route sync on | `cudaMemcpy 862.74 ms / 3168 calls`, `cudaDeviceSynchronize 25.28 ms / 165 calls` | DtoH `6.01 ms / 3399 calls` |

The no-readback ABI does remove the intended `cudaMemcpy` calls, but with the
current execution loop it exposes the same wait at later device synchronizes.
Skipping route validation readback is the measured useful default.

## Decision

Ship the route-validation readback removal as the production default.

Keep `ggml_turbomind_mul_mat_grouped_total_tokens()` and the DS4 wrapper as an
opt-in profiling path, but do not default it yet. The next optimization needs
to reduce or move the stage/layer synchronization points, not just change which
API call pays for the wait.

Artifacts:

- `logs/from-cluster/sprint100-tm-nosync/soak-8slot-production-default/summary.json`
- `logs/from-cluster/sprint100-tm-nosync/soak-8slot-oldabi-routeoff/summary.json`
- `logs/from-cluster/sprint100-tm-nosync/soak-8slot-default/summary.json`
- `logs/from-cluster/sprint100-tm-nosync/soak-8slot-rollback/summary.json`
- `logs/from-cluster/sprint100-tm-nosync/profile-4slot-production-default/nvprof.log`
- `logs/from-cluster/sprint100-tm-nosync/profile-4slot-default/nvprof.log`
- `logs/from-cluster/sprint100-tm-nosync/profile-4slot-rollback/nvprof.log`
