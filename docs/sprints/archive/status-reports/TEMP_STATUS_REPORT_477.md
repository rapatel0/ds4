# TEMP Status Report 477: TP/EP Correctness Gate

## Status

Sprint 477 is implemented and measured. We now have a one-command TP/EP
correctness gate for the current 32-slot / 256K serving shape.

## New Harness

```text
tools/ds4-v100-tp-ep-correctness-gate.py
```

It runs two selected-token HTTP profiles, compares the response artifacts, and
fails on profile errors, HTTP failures, VRAM admission failures, NCCL SYS graph
edges, or selected-token parity mismatches.

It also supports a faster developer mode:

```text
--mode self
```

That mode starts one server, sends `2 * requests`, and compares the first half
against the second half.

## Cluster Result

Artifact:

```text
/localpool/ds4/workspace/s477-correctness-default-s32-t1
```

Shape:

```text
32 slots
256K context
position 262080
8 selected-token requests per leg
1 generated token
natural CUDA order
default no-SYS NCCL ring
```

Topline:

| Metric | Control | Candidate |
|---|---:|---:|
| HTTP 200 | 8/8 | 8/8 |
| Tokens | 1 | 1 |
| Min free VRAM | 2086 MiB | 2086 MiB |
| NCCL graph SYS edges | 0 | 0 |
| Selected-token parity | 8/8 matched | 8/8 matched |
| Failed pairs | 0 | 0 |

## Rerun

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

## Decision

Use this before risky TP/EP performance work. It is not a full production
readiness test, but it makes the deterministic selected-token, VRAM, and no-SYS
checks trivial to repeat.

## Fast Iteration

Artifact:

```text
/localpool/ds4/workspace/s477-correctness-self-s32-t1-r4
```

Result:

| Metric | Value |
|---|---:|
| Mode | self |
| HTTP 200 | 8/8 |
| Matched pairs | 4/4 |
| Failed pairs | 0 |
| Min free VRAM | 2086 MiB |
| NCCL graph SYS edges | 0 |

Use `--mode self` for day-to-day iteration. Keep the default two-startup gate
for promotion checks because it catches startup and residency differences.
