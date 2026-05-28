# TEMP Status Report 470: Routed-FFN Suffix Replay Isolation

## Topline

Sprint 470 isolated the first persistent graph suffix slice at layer 0.
Routed-FFN-only suffix replay is correctness-clean in the direct diagnostic and
has a real isolated speed signal.

This does not make persistent graph serving production-ready. It narrows the
remaining graph correctness problem to the suffix stages after routed FFN:
dense overlap and compose/final-HC.

## Code Changes

- Added `--decode-cudagraph-suffix-stage-gate routed_ffn` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added `--decode-cudagraph-suffix-stage routed_ffn` profile plumbing.
- Fixed resident-profile direct diagnostics so deferred NCCL is opened before
  `--hc-current-nccl-allgather` runs.
- Added stable hash shortening for long profile artifact suffixes.

## V100 Evidence

Build:

```text
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Layer-0 direct diagnostic at `8` slots / `256K` / `3` decode steps:

| Mode | rc | Checksum | Capture | Replay | Decode ms/step | Slot-step tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Eager/control | 0 | `1510241683` | 0 | 0 | `35.897593` | `222.856165` |
| Persistent routed suffix | 0 | `1510241683` | 1 | 1 | `25.696161` | `311.330552` |

Persistent routed-suffix graph:

```text
nodes=9
replay_launches=3
instantiate_ms=0.211783
replay_ms=0.664672
```

Route shape:

```text
routes=0,16,0,0,0,0,16,16
active_experts=0,2,0,0,0,0,2,2
max_routes_per_expert=0,8,0,0,0,0,8,8
```

Artifacts:

```text
/localpool/ds4/workspace/logs/s470-routed-ffn-control-gpuroute
/localpool/ds4/workspace/logs/s470-routed-ffn-control3
/localpool/ds4/workspace/logs/s470-routed-ffn-persistent3
```

## Interpretation

Routed FFN is not the first unsafe persistent replay stage. The isolated speed
signal is about `1.40x`, but this is a layer-slice diagnostic, not full-server
tok/s.

Next work should split the remaining suffix into:

1. dense overlap only;
2. compose/final-HC only;
3. then full suffix replay;
4. then HTTP parity and throughput.
