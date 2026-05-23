# Sprint 283 - TP/EP FP16 Return Recheck

Date: 2026-05-23

## Goal

Recheck FP16 EP return under the promoted event-wait compose path to see
whether reducing peer-copy bytes now improves serving throughput.

## Implementation

Updated `tools/ds4-v100-run-appliance.sh`.

- Added `DS4_V100_TP_EP_RETURN_FP16`.
- Wires the setting to `--ep-return-fp16` in TP/EP serve mode.
- Records the setting in launcher config output and `startup.env`.
- Keeps the default off.

Updated `tools/ds4-v100-tp-ep-http-bench.sh`.

- Added `--ep-return-fp16`.
- Passes `DS4_V100_TP_EP_RETURN_FP16=1` into the launcher for A/B runs.

Updated `deploy/v100/ds4-v100-appliance.env.example`.

- Documents `DS4_V100_TP_EP_RETURN_FP16=0`.

## Validation

Local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-tp-ep-http-bench.sh
git diff --check
```

V100 same-binary A/B:

```text
ctx=262144
slots=32
tokens_per_request=64
generation_requests=3
copy_event_compose=on
```

Results:

| Mode | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | EP ms | Compose ms | Compose copy ms | Compose final ms | Match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 return | 766.883263 | 771.598469 | 997.165341 | 1004.292031 | 2588.558632 | 3571.085652 | 1770.953766 | 1271.423626 | 96/96 |
| FP16 return | 635.936079 | 639.624099 | 793.283316 | 797.595293 | 2616.186577 | 5127.088703 | 2085.464972 | 2351.603225 | 96/96 |

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint283-tp-ep-fp16-return/cluster/
```

Subdirectories:

- `event64_fp16/`
- `event64_fp32/`

Each contains the sustained HTTP summary, per-request responses,
`status_after.json`, `metrics.txt`, GPU utilization, and server logs.

## Decision

Reject FP16 EP return for the promoted serving path. It halves peer-copy
payload bytes in principle, but the extra cast/add/final-compose work dominates
on V100. Under the current event-wait compose path it regresses wall generated
throughput by `17.1%` and decode generated throughput by `20.4%`.

Keep the toggle available as a diagnostic, but keep
`DS4_V100_TP_EP_RETURN_FP16=0` as the appliance default.

## Next

- Continue compose optimization on the FP32 return path.
- Candidate next cuts:
  - reduce the number of staged FP32 contribution copies,
  - combine EP contribution reduction with destination compose more directly,
  - inspect whether the all-destination contribution layout is forcing
    avoidable traffic.
