# Sprint 074 Report: Async Peer Handoff Probe

## Summary

Sprint 074 shipped an opt-in async HC handoff path using queued device/device
and peer copies. The path is correct and slightly faster under the per-step
async pipeline, but the measured gain is below the threshold for changing the
appliance default.

## Implementation

- Added `ds4_gpu_tensor_copy_async`:
  - same-device uses `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, 0)`;
  - peer copy uses `cudaMemcpyPeerAsync(..., 0)` after setting the destination
    device.
- Added `ds4_v100_stage_scheduler_handoff_slot_span_async`.
- Added `ds4_v100_replay_options.async_handoff`.
- Reused the async handoff in serial, wavefront, per-step, persistent, and
  mailbox replay paths when explicitly enabled.
- Wired the flag through:
  - `tools/ds4-v100-replay --async-handoff`;
  - `tools/ds4-v100-sustained-decode-bench.sh --async-handoff`;
  - `DS4_V100_ASYNC_HANDOFF=1` in `tools/ds4-v100-run-appliance.sh`;
  - `tools/ds4-v100-appliance-soak.sh --async-handoff`.
- Status JSON reports `async_handoff`.

## V100 Validation

Build and shell checks:

```bash
bash -n tools/ds4-v100-sustained-decode-bench.sh
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-appliance-soak.sh
CUDA_ARCH=sm_70 make tools/ds4-v100-replay \
  tests/cuda_v100_stage_wavefront_smoke \
  tests/cuda_v100_selected_token_smoke
```

Correctness:

```text
cuda_v100_stage_wavefront_smoke: token0=16 token1=926 max_abs_slot0=0 max_abs_slot1=0 ok
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 ... expected=3136 ... ok
```

Short async-handoff smoke:

- context `262144`, slots `2`, tokens/request `2`;
- `2/2` token matches;
- benchmark metadata reports `async_handoff=1`.

Launcher/config:

- invalid async mode still exits with `rc=2`;
- launcher `--check` accepts `DS4_V100_ASYNC_HANDOFF=1` and reports
  `async_handoff=1`.

## Throughput Matrix

Fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- context: `1048576`
- slots: `2,4`
- queue policy: `sequential`
- async pipeline mode: `per-step`
- tokens/request: `16`
- measured requests/case: `4`
- warmup requests/case: `1`
- expected first token hex: `3136`

| Handoff | Slots | Generated tok/s | Continuation tok/s | Avg GPU util | Async total ms | Handoff sum ms | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocking | 2 | `5.553165` | `5.206092` | `14.739%` | `5594.042` | `43.673` | baseline |
| blocking | 4 | `8.605744` | `8.067885` | `19.885%` | `7083.395` | `263.624` | baseline |
| async | 2 | `5.591514` | `5.242044` | `14.633%` | `5558.764` | `38.839` | `+0.691%` |
| async | 4 | `8.738546` | `8.192387` | `19.025%` | `6977.006` | `179.154` | `+1.543%` |

## Decision

Async handoff is correct and modestly positive, but below the `3%` threshold for
changing the default. Keep appliance `auto` on per-step blocking handoff and
leave `--async-handoff` as opt-in. The next sprint should either:

- add explicit CUDA stream/event handoff instead of relying on default-stream
  ordering, or
- pivot to kernel-side work where the current profile shows larger absolute
  time.

## Artifacts

- `logs/from-cluster/sprint074-async-handoff-smoke`
- `logs/from-cluster/sprint074-perstep-blocking`
- `logs/from-cluster/sprint074-perstep-async-handoff`
- `logs/from-cluster/sprint074-handoff-comparison`

## Validation

- local C object compile for changed replay/scheduler files;
- shell syntax checks for changed scripts;
- V100 CUDA build for replay and smokes;
- V100 wavefront and selected-token smokes;
- V100 short async-handoff sustained smoke;
- V100 1M/2 and 1M/4 blocking-vs-async handoff matrix;
- JSON artifact validation;
- `git diff --check`.
