# TEMP Status Report 476: Direct Peer-Copy Topology Guard

## Status

Sprint 476 is implemented and measured. The NCCL topology guard is clean, but
direct peer copies are not yet no-SYS.

## What Changed

- Added `ds4_peer_copy_async` accounting around every direct
  `cudaMemcpyPeerAsync` in the TP/EP serving binary.
- Added diagnostic controls:
  - `DS4_V100_TP_EP_PEER_ACCOUNTING`
  - `DS4_V100_TP_EP_PEER_REJECT_SYS`
  - `--tp-peer-accounting`
  - `--tp-peer-reject-sys`
- Added `/status` and `/metrics` peer-copy counters so profile artifacts retain
  data even when the HTTP harness terminates the server.

## Cluster Result

Artifact:

```text
/localpool/ds4/workspace/s476-peer-account-status-s32-t2
```

Shape:

```text
32 slots
256K context
32 HTTP requests
2 generated tokens
NCCL no-SYS policy enabled
peer accounting enabled
peer reject disabled
```

Topline:

| Metric | Value |
|---|---:|
| HTTP 200 | 32/32 |
| Server generated decode tok/s | 37.778247 |
| Server continuation decode tok/s | 37.825860 |
| Min free VRAM | 2838 MiB |
| NCCL graph SYS edges | 0 |
| Direct peer-copy ops | 1,488,745 |
| Direct peer-copy bytes | 12.59 GiB |
| Direct SYS ops | 638,028 |
| Direct SYS bytes | 5.39 GiB |
| First SYS edge | src 0 -> dst 5, 3,072 bytes |

## Decision

Keep `DS4_V100_TP_EP_PEER_REJECT_SYS=0` by default. Promoting the rejection
guard would break the current serving path.

Next step: add call-site operation labels to peer-copy accounting, then replace
or reroute the highest-volume SYS-classified direct copy classes.
