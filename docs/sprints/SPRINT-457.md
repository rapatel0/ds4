# Sprint 457: Exclusive TP/EP HTTP A/B Harness Lock

## Objective

Prevent overlapping DS4 V100 TP/EP HTTP A/B runs from sharing the same 8-GPU
node and corrupting measurements with false OOMs, low utilization, or stale
server processes.

## Rationale

Sprint 456's first target-shape run failed before control startup with:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9723: out of memory
```

The root cause was not the candidate. An unrelated orphaned 4-token
`skip-slot-major` A/B was still resident on all GPUs. Existing cleanup is
port-scoped and per-case lock files are intentionally isolated, so two A/B
harnesses can run concurrently on different ports and oversubscribe VRAM.

Future graph/launch work needs clean measurement more than another ad hoc
process cleanup.

## Implementation

Add an A/B-level exclusive lock to:

```text
tools/ds4-v100-tp-ep-nccl-http-ab.py
```

Requirements:

- Default lock path is node-local and shared by all TP/EP A/B runs.
- `--no-global-lock` disables it for explicit diagnostics only.
- `--global-lock-file PATH` overrides the path.
- Lock acquisition is non-blocking by default so accidental overlap fails
  immediately before launching any GPU process.
- `--wait-global-lock` allows queued runs when intentionally desired.
- The lock is held for the full control+candidate A/B lifetime.
- The failure message identifies the lock path and suggests checking existing
  DS4 processes.

## Definition of Done

- Local Python syntax checks pass.
- Remote Python syntax checks pass.
- A no-GPU lock contention test proves a second harness instance fails before
  profile launch.
- Normal `--help` still works.
- `VISION.md` records this as a measurement hygiene promotion.

## Out Of Scope

- Killing arbitrary DS4 processes by default.
- Changing serving runtime lock semantics.
- Graph replay implementation.

## Outcome

Implemented in `tools/ds4-v100-tp-ep-nccl-http-ab.py`:

- Default global lock path:

  ```text
  /localpool/ds4/workspace/ds4-tp-ep-http-ab.lock
  ```

  when `/localpool/ds4/workspace` exists; otherwise:

  ```text
  /tmp/ds4-tp-ep-http-ab.lock
  ```

- `--global-lock-file PATH`
- `--no-global-lock`
- `--wait-global-lock`
- `--lock-check-only`

The lock is acquired before launching the control profile and is held for the
full A/B process lifetime.

Validation:

- Local `python3 -m py_compile tools/ds4-v100-tp-ep-nccl-http-ab.py`: pass
- Local `--help` exposes the lock controls: pass
- Local contention test: second harness exits `73` before profile launch
- Local free-lock check: pass
- Remote `python3 -m py_compile`: pass
- Remote `--help` exposes the lock controls: pass
- Remote contention test: second harness exits `73` before profile launch
- Remote free-lock check: pass
- Remote GPU state after validation: all 8 GPUs at `0 MiB` used

## Decision

Promote as permanent measurement hygiene. Future TP/EP HTTP A/B runs are
exclusive by default, which prevents the false OOM and polluted utilization
measurements seen during Sprint 456.
