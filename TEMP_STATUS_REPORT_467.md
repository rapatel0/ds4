# TEMP Status Report 467

## Focus

TP/EP graph-event-order correctness at `8` slots / `256K`.

## What Changed

- Added default-off HC-current sync gate.
- Added default-off per-stage graph sync gate:
  `DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC`.
- Added NCCL compatibility header for direct V100-node builds when `nccl.h` is
  missing but `libnccl.so` exists.
- Tested a graph-safe typed-KV event barrier and store-side system fences.

## Result

Minimal stage-level correctness barrier found:

```text
DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC=typed_history
```

This restored HTTP response parity:

```text
artifact: /localpool/ds4/workspace/logs/s467-stage-sync-typed-history-s8-t1
shape:    8 requests / 8 slots / 256K / 1 token
parity:   8/8 matched
server decode: control 20.619518 tok/s, candidate 9.358436 tok/s
```

HC-current-only sync failed, raw-read-only sync failed, and typed-KV event
barrier plus store-side `__threadfence_system()` still failed.

## Current Read

The graph corruption is localized to typed KV history visibility before raw
attention reads. A host stream sync after typed-history repairs correctness, but
an event barrier does not. This points at the peer-read typed KV row load path,
not the output head or routed FFN.

## Next

Implement a graph-safe typed KV history load that avoids immediate peer reads of
remote KV bytes, likely by local-shard load plus explicit NCCL/peer row assembly.
