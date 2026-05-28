# TEMP Status Report 474: Locked Steady-State Profile Recovery

## Recovery State

After the local interruption/power event, the V100 node is reachable and idle:

```text
host=gpu-01
uptime > 1 day
GPU memory used: 0 MiB on all 8 GPUs
DS4 processes: none
global lock owner: none
```

The completed S474 artifact survived and is valid:

```text
/localpool/ds4/workspace/logs/s474-steady-profile-s32-r256-t64-c32d-lock
profile_returncode=0
```

## What Changed

- Added a permanent locked steady-state profiler:
  `tools/ds4-v100-tp-ep-steady-profile.py`.
- Added `--request-concurrency` to the HTTP profile harness.
- Added steady-profile participation in the global benchmark lock:
  `/localpool/ds4/workspace/ds4-tp-ep-http-ab.lock`.
- Fixed dmon capture to include PCIe throughput via `pucmt`.
- Fixed the idle detector so observer commands do not block queued profiles.

## Clean S474 Result

Shape:

```text
ctx=262144
slots=32
requests=256
tokens=64
request_concurrency=32
position=262000
```

Topline:

| Metric | Value |
|---|---:|
| HTTP 200 | 256/256 |
| Client generated tok/s | 15.415121 |
| Server generated decode tok/s | 35.899801 |
| Server continuation decode tok/s | 35.898426 |
| Min free VRAM | 1734 MiB |
| dmon SM util avg | 10.842227% |

Key timers:

| Timer | ms |
|---|---:|
| Decode domain | 889.500863 |
| HC-current input | 356.678930 |
| EP | 473.516307 |
| Compose | 18.781809 |
| Attention projection | 53.247597 |
| Compressed KV | 53.698680 |
| Post-attention FFN input | 55.457515 |

DCGMI:

| Window | GPU util avg | Main signal |
|---|---:|---:|
| A.1 SM/occupancy | 11.326042% | `sm_active=0.025557`, `sm_occupancy=0.012456` |
| A.2 tensor | 12.333333% | `tensor_active=0.001227` |
| A.4 fp32 | 12.177083% | `fp32_active=0.004965` |
| A.5 fp16 | 11.530208% | `fp16_active=0.000781` |

## Interpretation

This confirms the model is not bottlenecked on tensor-core math in the current
serving shape. Tensor activity is near zero, DRAM activity is low, and decode
time remains dominated by HC-current/post-attention staging and EP orchestration.

The next useful implementation target is not another PP variant or a narrow
dtype swap. It is a TP/EP rank-major/rank-local staging reduction around the
current-hidden and post-attention boundaries, followed by another locked
steady-state profile.
