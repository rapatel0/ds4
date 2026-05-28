# TEMP Status Report 457

## Current Focus

Adding an exclusive node-level lock to the TP/EP HTTP A/B harness so future
throughput experiments cannot overlap on the same 8-GPU V100 node.

## Target

```text
tool: tools/ds4-v100-tp-ep-nccl-http-ab.py
default lock: /localpool/ds4/workspace/ds4-tp-ep-http-ab.lock when available
fallback:     /tmp/ds4-tp-ep-http-ab.lock
```

## Validation Plan

- `python3 -m py_compile` locally and remotely.
- `--help` locally and remotely.
- Hold the lock in one process and verify the harness exits before launching
  profile subprocesses when a second process attempts to acquire it.

## Result

Implemented in `tools/ds4-v100-tp-ep-nccl-http-ab.py`:

```text
--global-lock-file PATH
--no-global-lock
--wait-global-lock
--lock-check-only
```

Default lock path is `/localpool/ds4/workspace/ds4-tp-ep-http-ab.lock` on the
V100 node, with `/tmp/ds4-tp-ep-http-ab.lock` as local fallback.

Validation passed:

```text
local py_compile:       pass
local help:             pass
local contention:       rc=73 before profile launch
local free-lock check:  pass
remote py_compile:      pass
remote help:            pass
remote contention:      rc=73 before profile launch
remote free-lock check: pass
cluster after check:    0 MiB used on all 8 GPUs
```

This does not improve tok/s directly, but it removes a repeated source of
invalid A/Bs and false OOMs before returning to graph/launch work.
