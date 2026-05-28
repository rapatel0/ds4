# TEMP Status Report 427

## Focus

Sprint 427 added a direct parity audit for the rank-major FFN half inputs.

The point was to stop inferring correctness from final checksums and compare
the exact `__half` buffers before cuBLAS/TurboMind consumes them.

## Implementation

Added default-off gate:

```text
--routed-ffn-rank-major-input-parity-gate
```

It emits:

```text
tp_ep_rank_major_half_input_diff
```

Compared buffers:

- rank-major shared gate input vs legacy slot-major half conversion
- rank-major shared up input vs legacy slot-major half conversion
- rank-major routed `r.d_a` vs legacy route-slot pack

## Build

V100 sm_70 build passed for:

```text
tools/ds4-v100-tp-ep-full-layer-smoke
```

## Evidence

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint427-rankmajor-half-input-parity-syncplan/
```

Shape:

```text
slots=8
ctx=262144
decode_steps=1
tp_runtime_scratch_mib=128
HC-current NCCL allgather=on
post-attention FFN input=on
CUDA graph replay=off
route-plan async upload=off
```

Direct parity result:

| Case | Diff lines | Mismatch lines | Result |
|---|---:|---:|---|
| shared-only | 688 | 0 | clean |
| route-only | 329 | 0 | clean |

Same-mode checksums:

| Case | Decode tok/s | Checksum |
|---|---:|---:|
| control | 12.525670 | 8358757728 |
| shared-only + parity | 10.650018 | 8358757728 |
| route-only + parity | 12.356142 | 8358757728 |

## Interpretation

The rank-major shared and routed half-input kernels are not the blocker in the
synchronous-plan eager regime. They produce byte-identical half buffers, and the
all-layer checksum matches control.

The previous divergence therefore points at the persistent-graph /
async-route-plan regime, not the half-input values themselves.

## Next

Sprint 428 should isolate graph/async route-plan behavior:

- compare route metadata between sync upload and async upload
- add graph-safe device-resident audit counters if needed
- rerun rank-major shared/route split in the exact persistent-graph regime
  that diverged in Sprint 425
