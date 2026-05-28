# Sprint 470: Layer-0 Suffix Replay Isolation

## Objective

Find the first graph-unsafe stage inside the persistent replay suffix without
running full HTTP A/Bs for every guess.

## Rationale

Sprint 469 showed persistent replay has a real speed signal but fails response
parity from layer 0. Moving dynamic HC/current, attention/KV, and route prep out
of the graph did not fix parity, and adding a prefix completion barrier did not
fix it either. The remaining captured suffix must be split into smaller replay
units.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add a direct layer-0 diagnostic path for replaying suffix stages one at a
  time:
  - routed FFN only;
  - dense overlap only;
  - compose/final-HC only.
- Use checksums before full HTTP parity.

## Definition of Done

- A V100-buildable diagnostic mode exists for suffix-stage replay isolation.
- At least one stage-isolation run completes on the V100 node.
- The sprint records whether the first unsafe replay stage is routed FFN, dense,
  or compose/final-HC.

## Implementation

- Added `--decode-cudagraph-suffix-stage-gate routed_ffn` to the direct
  TP/EP layer diagnostic.
- Added profile wrapper plumbing through
  `--decode-cudagraph-suffix-stage routed_ffn`.
- Fixed the resident-profile direct path so deferred NCCL communicators are
  opened before `--hc-current-nccl-allgather` runs.
- Shortened long profiler artifact suffixes with a stable hash when needed, so
  persistent graph variants no longer exceed filesystem filename limits.

## Validation

V100 build:

```text
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Initial control findings:

- The first routed-only control failed before the suffix at HC-current allgather:
  deferred NCCL had not been opened in the resident-profile branch.
- After fixing deferred NCCL, the next control failed because the profile
  defaulted to async host route upload while routed-normalized input requires
  the GPU route planner.
- With `--gpu-route-plan`, the layer-0 routed-FFN suffix control passed.

Layer-0 routed-FFN suffix A/B at `8` slots / `256K` / `3` decode steps:

| Mode | rc | Checksum | Capture | Replay | Decode ms/step | Slot-step tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Eager/control | 0 | `1510241683` | 0 | 0 | `35.897593` | `222.856165` |
| Persistent routed suffix | 0 | `1510241683` | 1 | 1 | `25.696161` | `311.330552` |

Persistent routed-suffix capture details:

```text
nodes=9
replay_launches=3
instantiate_ms=0.211783
replay_ms=0.664672
```

Route shape for this deterministic layer-0 diagnostic:

```text
routes=0,16,0,0,0,0,16,16
active_experts=0,2,0,0,0,0,2,2
max_routes_per_expert=0,8,0,0,0,0,8,8
```

Remote artifacts:

```text
/localpool/ds4/workspace/logs/s470-routed-ffn-control-gpuroute
/localpool/ds4/workspace/logs/s470-routed-ffn-control3
/localpool/ds4/workspace/logs/s470-routed-ffn-persistent3
```

## Outcome

Routed FFN suffix replay is graph-safe for the layer-0 direct diagnostic:
checksum matches eager and replay improves the isolated layer slice by about
`1.40x`.

This does not yet promote persistent graph serving. It only removes routed FFN
as the first suspected unsafe suffix stage. The next split should isolate the
remaining dense-overlap and compose/final-HC suffix stages before another full
HTTP graph A/B.
