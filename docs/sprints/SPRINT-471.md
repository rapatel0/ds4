# Sprint 471: DCGMI Profiling Metrology Correction

## Objective

Replace misleading `nvidia-smi dmon` GPM-based metrology with a V100-correct
DCGMI sampler in the TP/EP profile harness and use it for a clean serving
measurement.

## Rationale

The V100 node exposes fine-grained profiling counters through `dcgmi dmon`, not
through the broad `nvidia-smi dmon --gpm-metrics` path used in earlier
experiments. The updated `gpu-profiling-guidance.md` also makes the important
counter-group rule explicit: `1004 tensor_active` is A.2 and must not be mixed
with A.1 `1002/1003` occupancy fields in the same performance pass.

## Scope

- TP/EP profiling harness only.
- No PP/layer-split work.
- No kernel optimization in this sprint.
- Add a `dcgmi` GPU sampler backend.
- Keep `nvidia-smi dmon` available for cheap NVML health telemetry.
- Default DCGMI fields to the V100 zero-multiplex set:
  `203,252,155,150,1002,1003,1005,1009,1010,1001,1011,1012`.
- Leave tensor-active `1004` available through `--dcgmi-fields`, but collect it
  in a separate pass.

## Definition of Done

- `tools/ds4-v100-tp-ep-profile.py` supports `--gpu-sampler dcgmi`.
- `tools/ds4-v100-tp-ep-nccl-http-ab.py` forwards `--gpu-sampler dcgmi` and
  `--dcgmi-fields` into both A/B legs.
- DCGMI samples are timestamped and parsed into `summary.json`.
- Summary includes SM active, SM occupancy, DRAM active, PCIe bytes, NVLink
  bytes, power, temperature, and VRAM where fields are present.
- `nvidia-smi dmon` no longer requests broad GPM fields by default.
- Local and remote Python validation pass.
- A full DS4 serving run reaches `responses_complete` and writes `summary.json`
  using the corrected DCGMI sampler.
- A separate tensor-active serving pass reaches `responses_complete`.

## Outcome

Implemented in `tools/ds4-v100-tp-ep-profile.py` and forwarded through
`tools/ds4-v100-tp-ep-nccl-http-ab.py`.

Validation:

- Local py_compile:
  `tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py` pass.
- Remote py_compile on gpu-01: pass.
- Remote sampler-only DCGMI smoke:
  `/localpool/ds4/workspace/logs/dcgmi-sampler-smoke/gpu_util.csv`
  - `gpu_sample_source=dcgmi`
  - `gpu_sample_count=32`
  - numeric `sm_active`, `sm_occupancy`, `dram_active`, `nvlink_tx_bytes`,
    `nvlink_rx_bytes`, `power_w`, and `fb_used_mib` fields.

Clean serving evidence:

- Default zero-multiplex pass:
  `/localpool/ds4/workspace/logs/s470-dcgmi-serving-s8-t8-default/...`
  - `responses_complete` reached, `http_200=8/8`, `summary.json` written.
  - `client_generated_tok_s=3.667873`.
  - `gpu_sample_source=dcgmi`, `gpu_sample_count=1968`,
    `gpu_steady_sample_count=280`.
  - Request-window `gpu_steady_util_avg=8.871429%`.
  - Request-window `sm_active_avg=0.016914`, `sm_occupancy_avg=0.005125`,
    `dram_active_avg=0.005089`, `gr_engine_active_avg=0.049082`.
  - Request-window NVLink averages were symmetric:
    `nvlink_tx_bytes_avg=60513147.99`, `nvlink_rx_bytes_avg=60517512.20`.
- Separate tensor-active pass:
  `/localpool/ds4/workspace/logs/s470-dcgmi-serving-s8-t8-tensor/...`
  - `responses_complete` reached, `http_200=8/8`, `summary.json` written.
  - `client_generated_tok_s=3.465720`.
  - `gpu_sample_source=dcgmi`, `gpu_sample_count=2024`,
    `gpu_steady_sample_count=288`.
  - Request-window `tensor_active_avg=0.000694`, `tensor_active_max=0.002`.

## Decision

Promote the DCGMI sampler as the profiling path for TP/EP performance work on
gpu-01. Do not use broad `nvidia-smi dmon --gpm-metrics` results for V100
performance decisions.

The corrected counters reinforce the current bottleneck hypothesis: this
serving shape is not lighting up Volta tensor cores. SM activity and occupancy
are low, tensor activity is effectively zero, and the path is dominated by
launch/control/staging or non-HMMA work rather than saturated GEMM.

Next performance work should use these counters to verify whether each proposed
change increases request-window `tensor_active` and `sm_occupancy`, not just
NVML GPU utilization.
