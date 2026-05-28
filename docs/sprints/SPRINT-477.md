# Sprint 477: TP/EP Correctness Gate Harness

## Overview

This sprint adds a repeatable correctness gate for the TP/EP appliance so
future optimization work can be rerun without reconstructing the test command
from terminal history.

The gate is intentionally TP/EP-only. No PP or layer-split path is in scope.

## Goal

Create a single command that launches the current TP/EP HTTP serving harness,
runs deterministic selected-token requests twice at the same shape, and fails
if any of the correctness or admission invariants are broken.

## Implementation

Added:

```text
tools/ds4-v100-tp-ep-correctness-gate.py
```

The harness runs a control profile and a candidate profile back-to-back, then
compares their response artifacts with:

```text
tools/ds4-v100-http-response-parity.py --ignore-text
```

The gate checks:

- both profile runs exit cleanly
- every selected-token HTTP request returns 200
- generated token count matches the requested shape
- VRAM admission reports no failures
- minimum free VRAM stays above the configured threshold
- NCCL graph dump has zero SYS edges when present
- optional direct peer-copy accounting reports zero SYS ops when requested
- selected-token response artifacts match deterministically

The script writes:

```text
control-command.txt
candidate-command.txt
control.log
candidate.log
response-parity.json
response-parity.log
correctness-summary.json
correctness-summary.md
```

Modes:

- `--mode two-run` starts isolated control and candidate servers. This is the
  promotion-grade check because it catches startup and residency differences.
- `--mode self` starts one server, sends `2 * requests`, and compares the first
  half against the second half. This is the faster developer smoke because it
  pays the model residency load once.

## Validation

Local validation:

```bash
python3 -m py_compile \
  tools/ds4-v100-tp-ep-correctness-gate.py \
  tools/ds4-v100-tp-ep-profile.py \
  tools/ds4-v100-http-response-parity.py \
  tools/ds4-v100-http-readiness-check.py

git diff --check -- tools/ds4-v100-tp-ep-correctness-gate.py
```

Cluster artifact:

```text
/localpool/ds4/workspace/s477-correctness-default-s32-t1
```

Rerun command:

```bash
cd /localpool/ds4/workspace/ds4-sprint181
rm -rf /localpool/ds4/workspace/s477-correctness-default-s32-t1
python3 tools/ds4-v100-tp-ep-correctness-gate.py \
  --repo-dir . \
  --artifact-dir /localpool/ds4/workspace/s477-correctness-default-s32-t1 \
  --ctx 262144 \
  --slots 32 \
  --experimental-ctx-slot-cap 32 \
  --position 262080 \
  --tokens 1 \
  --requests 8 \
  --max-requests 16 \
  --request-concurrency 8 \
  --port-base 19100 \
  --wait-global-lock
```

Result:

| Check | Value |
|---|---:|
| Gate passed | true |
| Control HTTP 200 | 8/8 |
| Candidate HTTP 200 | 8/8 |
| Matched selected-token pairs | 8/8 |
| Failed selected-token pairs | 0 |
| Control min free VRAM | 2086 MiB |
| Candidate min free VRAM | 2086 MiB |
| Control NCCL SYS graph edges | 0 |
| Candidate NCCL SYS graph edges | 0 |

Startup observation:

```text
tp_ep_shared_expert_bindings_load bytes=147169738752 load_ms=78723.310476
tp_ep_shared_expert_bindings_load bytes=147169738752 load_ms=78346.938809
```

The two isolated server startups make the gate clean but slow. `--mode self`
reuses one loaded server for self-parity, while the two-startup mode remains
the promotion-grade gate.

Fast self-mode validation:

```text
artifact: /localpool/ds4/workspace/s477-correctness-self-s32-t1-r4
mode:     self
shape:    32 slots, 256K context, position 262080, 4 pairs
result:   passed, 4/4 matched, 0 failed, HTTP 200 8/8, NCCL SYS edges 0
```

Rerun command:

```bash
cd /localpool/ds4/workspace/ds4-sprint181
rm -rf /localpool/ds4/workspace/s477-correctness-self-s32-t1-r4
python3 tools/ds4-v100-tp-ep-correctness-gate.py \
  --mode self \
  --repo-dir . \
  --artifact-dir /localpool/ds4/workspace/s477-correctness-self-s32-t1-r4 \
  --ctx 262144 \
  --slots 32 \
  --experimental-ctx-slot-cap 32 \
  --position 262080 \
  --tokens 1 \
  --requests 4 \
  --max-requests 8 \
  --request-concurrency 8 \
  --port-base 19120 \
  --wait-global-lock
```

## Decision

Promote `tools/ds4-v100-tp-ep-correctness-gate.py` as the default correctness
gate before risky performance work. The current TP/EP selected-token path is
deterministic at the target `32` slot / `256K` context shape for this short
gate.

Do not treat this as full response parity or production readiness. It proves
the deterministic selected-token invariant, VRAM admission, and NCCL no-SYS
policy for the current serving path.

## Next

- Use `--mode self` during local iteration to avoid repeating the 147 GB
  residency load.
- Rebuild and rerun with `--tp-peer-accounting` once the remote binary includes
  the latest peer-copy CLI flags.
- Use this gate as the first check before rewriting HC-current staging or
  direct SYS peer-copy routing.
