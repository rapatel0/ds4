# Sprint 425: Split Rank-Major FFN Input Probes

## Objective

Isolate the all-layer checksum divergence from
`--routed-ffn-rank-major-input-gate` by splitting the combined post-attention
FFN input rewrite into shared-only and routed-only diagnostic gates.

No PP/layer-split work is in scope.

## Context

Sprint 424 showed:

- Resident layers 0, 1, and 2 match with the combined rank-major FFN input gate.
- All-layer persistent graph runs still diverge.
- Splitting post-attention rank-major scratch from HC-current scratch did not
  restore parity.
- Serial EP/dense did not prove an overlap-only issue.

The next useful question is whether divergence is caused by shared FFN gate/up
input, routed expert `r.d_a` input, or a state interaction when both are active.

## Implementation

Add default-off diagnostic gates:

```text
--routed-ffn-rank-major-shared-input-gate
--routed-ffn-rank-major-route-input-gate
```

Keep existing behavior:

```text
--routed-ffn-rank-major-input-gate
```

as the combined gate that enables both split gates.

When only one split gate is active, the other input family must continue to use
the existing slot-major `hc->d_ffn_normed` path.

## Definition of Done

- V100 sm_70 build passes.
- Existing combined resident layer parity still passes for at least layer 1.
- All-layer direct control, shared-only, route-only, and combined probes run at
  `8` slots / `256K` / `1` decode step.
- The sprint records the first differing layer for each diagnostic gate.
- Promotion remains blocked unless a split gate preserves all-layer checksum
  and improves timing.

## Outcome

V100 sm_70 build passed after adding the split gates.

The first implementation accidentally let the rank-major path replace the
common slot-major `hc->d_current_full` / RMSNorm / router-input path. That made
all split variants diverge immediately at layer 0 with the same checksum, so it
was not a clean shared-vs-routed isolation.

I corrected the diagnostic so the control path still builds `hc->d_current_full`
with the legacy slot-major gather and uses the same RMSNorm/router route
selection. The rank-major allgather is now only consumed by the selected
rank-major FFN half-input family.

Final all-layer diagnostic shape:

```text
slots=8
ctx=262144
decode_steps=1
tp_runtime_scratch_mib=128
persistent graph replay=on
deferred NCCL=on
HC-current NCCL allgather=on
post-attention FFN input=on
semantic stats skip=on
```

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint425-rankmajor-split-legacy-norm-s128/
```

| Case | Flags | Decode tok/s | Decode ms | Final checksum | Matching layer items | First diff |
|---|---|---:|---:|---:|---:|---|
| control | none | 47.954160 | 166.825984 | 4439536078 | 43/43 | - |
| shared-only | `--routed-ffn-rank-major-shared-input-gate` | 45.863077 | 174.432257 | 4112542066 | 0/43 | step 0 layer 0 |
| route-only | `--routed-ffn-rank-major-route-input-gate` | 45.677510 | 175.140895 | 4300822684 | 1/43 | step 0 layer 1 |
| combined | `--routed-ffn-rank-major-input-gate` | 47.236263 | 169.361408 | 751558149 | 0/43 | step 0 layer 0 |

First differing item details:

| Case | Control checksum | Candidate checksum | Control ms | Candidate ms |
|---|---:|---:|---:|---:|
| shared-only layer 0 | 511287928 | 4388409773 | 4.748288 | 5.796864 |
| route-only layer 1 | 4371684951 | 409095477 | 3.046400 | 4.760576 |
| combined layer 0 | 511287928 | 4388409773 | 4.748288 | 5.467136 |

The `256 MiB` scratch rerun hit CUDA OOM during all-layer expert residency when
the rank-major candidate duplicated FFN norm weights on the rank-local devices.
Reducing scratch to `128 MiB` kept the same 8-slot/256K diagnostic shape
admitted and produced the split evidence above.

Resident layer parity was not rerun as a decisive gate in this sprint. The
single-layer resident command returned rc `14` despite printing its internal
decode-loop PASS, so the all-layer persistent-graph checksum rows are the
authoritative result.

## Decision

Do not promote any rank-major FFN input gate.

The shared rank-major gate/up half-input path is the first blocker: it diverges
at layer 0 even when router selection remains on the legacy slot-major path.
The routed rank-major route-input path is a separate second blocker: it
preserves layer 0 but diverges at layer 1.

Next work should add direct parity instrumentation for:

- legacy `hc->d_ffn_normed` versus rank-major shared gate/up half input
- legacy `r.d_a` versus rank-major route packed half input
- per-layer route slots/weights under the route-only gate
