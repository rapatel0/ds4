# Sprint 474: Locked Steady-State DCGMI Profile

## Objective

Make TP/EP serving metrology reliable after the profiling false starts: collect
a long 32-slot / 256K steady-state decode profile with non-overlapping V100
DCGMI counter windows, continuous `nvidia-smi dmon` health telemetry, and the
same global benchmark lock used by HTTP A/B runs.

## Scope

- TP/EP only.
- No PP/layer-split work.
- No serving default performance promotion.
- Keep the target shape at `32` slots and `256K` context.
- Use the current router+FFN rank-major baseline with scratch `1280 MiB`.

## Implementation

- Added `tools/ds4-v100-tp-ep-steady-profile.py`.
- The steady profiler:
  - acquires `/localpool/ds4/workspace/ds4-tp-ep-http-ab.lock`;
  - waits for the node to be idle;
  - starts external `nvidia-smi dmon -s pucmt` before server launch;
  - waits for HTTP `requests_start`;
  - warms the request window before measuring counters;
  - records separate one-minute DCGMI windows for V100 conflicting groups:
    `1002/1003`, `1004`, `1006`, `1007`, and `1008`;
  - writes `steady-summary.json`.
- Added bounded request concurrency to
  `tools/ds4-v100-tp-ep-profile.py` so long runs can keep 32 active slots busy
  without opening one HTTP connection per queued request.
- Corrected the idle detector so observation commands do not block the queued
  steady profile.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-steady-profile.py tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
git diff --check -- tools/ds4-v100-tp-ep-steady-profile.py tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
PASS
```

Remote V100:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-steady-profile.py tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
PASS
```

Clean locked artifact:

```text
/localpool/ds4/workspace/logs/s474-steady-profile-s32-r256-t64-c32d-lock
```

Run shape:

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
| Coalesced batch size | 32 |
| Generated tokens in first response metadata | 2048 |
| Min free VRAM | 1734 MiB |
| Max used VRAM | 30759 MiB |
| Full dmon SM util avg | 10.842227% |

DCGMI one-minute windows:

| Window | GPU util avg | SM active avg | SM occupancy avg | Tensor/FP avg | DRAM active avg | NVLink TX avg | NVLink RX avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| A.1 `1002,1003` | 11.326042% | 0.025557 | 0.012456 | n/a | 0.008980 | 114.3 MB/sample | 115.0 MB/sample |
| A.2 `1004` | 12.333333% | n/a | n/a | tensor 0.001227 | 0.008347 | 132.3 MB/sample | 132.2 MB/sample |
| A.3 `1006` | 11.435417% | n/a | n/a | fp64 0.000000 | 0.008569 | 117.7 MB/sample | 117.7 MB/sample |
| A.4 `1007` | 12.177083% | n/a | n/a | fp32 0.004965 | 0.008898 | 127.9 MB/sample | 127.8 MB/sample |
| A.5 `1008` | 11.530208% | n/a | n/a | fp16 0.000781 | 0.009044 | 120.9 MB/sample | 120.8 MB/sample |

Stage timers from the completed profile:

| Timer | ms |
|---|---:|
| Decode domain | 889.500863 |
| HC-current input | 356.678930 |
| EP | 473.516307 |
| Compose | 18.781809 |
| Attention projection | 53.247597 |
| Compressed KV | 53.698680 |
| Post-attention FFN input | 55.457515 |

## Outcome

The previous low-utilization conclusion is now supported by a completed,
non-overlapping steady-state run. The model is not tensor-core saturated:
`tensor_active` averages about `0.0012`, `fp16_active` about `0.0008`, and
`fp32_active` about `0.0050` during separate DCGMI windows. DRAM activity is
also low. The dominant serving cost is still orchestration/staging around
HC-current/post-attention plus EP work, not raw HMMA throughput.

The permanent metrology improvement is promoted. Future long profiles should
use the locked steady profiler, or the HTTP A/B global lock, so overlapping
runs cannot pollute utilization, OOM, or throughput data.

## Next

1. Use the S474 evidence to target the largest remaining serving boundary:
   `hc_current_input` plus post-attention staging.
2. Keep persistent graph serving default-off until 32-slot parity is strict.
3. Continue TP/EP-only implementation; no PP/layer-split variants.
4. After the next implementation change, rerun the locked steady profile and
   compare server decode, DCGMI tensor activity, and stage timers.
