# TEMP Status Report 442

Date: 2026-05-27

## Focus

TP/EP only. Tested whether an actual-route routed-FFN executor is still the
right next performance lever.

## What Changed

- Added `DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC`.
- Added profile support for `--post-attention-device-actual-route-sync`.
- Added HTTP A/B harness support for
  `--candidate-post-attention-device-actual-route-sync`.
- Added Sprint 442 documentation and updated the vision.

## V100 Result

Artifact:

```text
/localpool/ds4/workspace/logs/s442-ab
```

Shape:

```text
8 requests, 8 slots, 256K context, 2 generated tokens
```

Result:

```text
control:   fixed-capacity route planner + HC-current NCCL
candidate: control + actual-route-sync
readiness: true / true
parity:    8/8 matched
first token: 72960 both legs

server generated decode tok/s:     14.080773 -> 13.885178
server continuation decode tok/s:  14.129698 -> 13.895648
client generated tok/s:             1.205039 -> 1.210999
avg GPU util:                       8.489% -> 8.571%
HC-current input ms:              384.282571 -> 389.666722
```

## Decision

Do not build the graph-safe active-route executor next.

The upper-bound diagnostic did not show material headroom. The next bottleneck
is above the routed executor: HC-current/post-attention staging, router/route
upload, and graph-serving promotion.

## Hygiene

- Local `python3 -m py_compile` passed for the touched Python harnesses.
- Local `bash -n tools/ds4-v100-run-appliance.sh` passed.
- Local `git diff --check` passed.
- Remote Python/shell syntax checks passed.
- No stale GPU compute processes remained after the A/B run.
