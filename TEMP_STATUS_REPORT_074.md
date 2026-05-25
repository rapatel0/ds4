# TEMP Status Report 074 - Sprint 362 Profile Defaults

Date: 2026-05-25

## Current Focus

TP/EP serving metrology hygiene. Sprint 362 aligns the permanent profile
harness with the production launcher default for fused compressed pool+norm.

## What Changed

`tools/ds4-v100-tp-ep-profile.py` now treats compressed pool+norm as a
tri-state in HTTP launcher mode:

- `--fused-compressed-pool-norm`: force env var to `1`.
- `--disable-fused-compressed-pool-norm`: force env var to `0`.
- neither flag: omit the env var and let `tools/ds4-v100-run-appliance.sh`
  supply the production default.

The two flags are mutually exclusive.

## V100 Proof

Shape:

```text
endpoint: /v100/selected-token
requests: 1
tokens/request: 1
slots: 32
context: 256K
position: 262143
```

| Harness mode | HTTP 200 | Command has pool gate | Fused pool layers |
|---|---:|---:|---:|
| default | 1/1 | true | 40 |
| disabled | 1/1 | false | 0 |

## Interpretation

- The profile harness default now matches the launcher default.
- The explicit disable flag gives a clean control path for future A/B tests.
- This sprint does not claim a throughput improvement; it prevents future
  profile runs from accidentally testing a different configuration than the
  production launcher.

## Next Best Step

Return to TP/EP implementation work. The strongest current candidates are:

1. deeper compressed-KV state/emit fusion, or
2. longer decode-heavy chat/selected-token serving runs after that fusion.

Artifacts:

```text
logs/from-cluster/sprint362-profile-launcher-defaults/
```
