# TEMP Status Report 425

## Focus

Sprint 425 split `--routed-ffn-rank-major-input-gate` into shared-only and
route-only diagnostics so we can stop treating the rank-major FFN input path as
one opaque failure.

## Implementation

Added:

```text
--routed-ffn-rank-major-shared-input-gate
--routed-ffn-rank-major-route-input-gate
```

The existing combined gate still enables both.

I also corrected the diagnostic after the first A/B showed that the rank-major
path was changing the common slot-major norm/router path. The current code keeps
legacy slot-major `hc->d_current_full`, RMSNorm, and router selection intact.
Rank-major allgather is used only by the selected FFN half-input path.

## V100 Build

V100 sm_70 build passed:

```text
tools/ds4-v100-tp-ep-full-layer-smoke
```

## Final Evidence

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint425-rankmajor-split-legacy-norm-s128/
```

Shape:

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

| Case | Decode tok/s | Decode ms | Final checksum | First diff |
|---|---:|---:|---:|---|
| control | 47.954160 | 166.825984 | 4439536078 | - |
| shared-only | 45.863077 | 174.432257 | 4112542066 | step 0 layer 0 |
| route-only | 45.677510 | 175.140895 | 4300822684 | step 0 layer 1 |
| combined | 47.236263 | 169.361408 | 751558149 | step 0 layer 0 |

First differing checksums:

| Case | Control | Candidate |
|---|---:|---:|
| shared-only layer 0 | 511287928 | 4388409773 |
| route-only layer 1 | 4371684951 | 409095477 |
| combined layer 0 | 511287928 | 4388409773 |

## Interpretation

The shared rank-major gate/up half-input path is the first blocker. It diverges
immediately at layer 0.

The routed rank-major route-input path is not clean either, but it gets past
layer 0 and first diverges at layer 1.

The combined gate follows the shared-only failure signature, so shared-input
parity should be fixed before spending more time on combined performance.

## Decision

Keep all rank-major FFN input gates default-off.

Next target: add parity probes that compare legacy half inputs against the
rank-major-produced half inputs before TurboMind/cuBLAS consumes them.
