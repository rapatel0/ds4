# TEMP Status Report 060 - HC Current Peer Gather

Date: 2026-05-25

## Current Focus

TP/EP-only serving optimization after Sprint 347 made direct CUDA profiling
usable. The bottleneck under test is `run_shared_hc_current_input`.

## Change

Added an opt-in peer-gather diagnostic path:

```text
--tp-hc-current-input-peer-gather-gate
DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER=1
tools/ds4-v100-tp-ep-profile.py --hc-current-peer-gather
```

The new CUDA kernel lets each rank build its own full current vector from the
eight current shards, then skips the old GPU0 full-current broadcast back to
all ranks.

## V100 Result

Shape:

```text
slots: 32
ctx: 262144
decode steps: 2
typed KV: history + skip-current-load + quiet + batch-rows + stream-sync
```

| Case | Generated tok/s decode | Continuation tok/s decode | sum decode ms | HC-current ms | Output finite |
|---|---:|---:|---:|---:|---:|
| Control | `87.263615` | `100.446187` | `733.409911` | `596.248809` | `0` bad |
| Peer gather | `67.495350` | `80.223389` | `948.213473` | `801.525057` | `0` bad |

## Decision

Rejected for promotion. The peer-gather path is correct for the tested window,
but slower. This says the next lever is not simply removing the GPU0
full-current broadcast. The better next target is reducing HC control
synchronization and fusing or bypassing the split/norm/fill chain.

## Artifacts

```text
logs/from-cluster/sprint348-hc-peer-gather/cluster/
```
